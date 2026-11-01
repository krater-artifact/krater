// RackCache (Krater) prototype. Hosts are the threads of one process (threads mode), or one process per host on a POSIX
// shm area (Config::shm_name, local_host).
//
// Read primitive: the fast path reads every object line with ONE aligned 64 B load; the slow path flushes the object's
// lines (clflushopt + mfence) BEFORE installing the local-index entry; every flush is followed by mfence; an AI entry is
// one 16 B item written with one store and read with one load.
//
// Shared area (would be CXL): per owner a slab of 64 B lines [gen 4B | payload 60B], an authoritative index
// (AI, 16 B entries {off | size class, gen, kg, key fingerprint}, open addressing) and one outgoing SPSC ring
// per receiver. Only the owner writes its slab and AI; only the sender writes a ring (the receiver writes its consumer
// counter in its own area). Shared structures hold offsets, never pointers.
// Host-local (would be each host's DRAM): local index LI + tombstones, owner metadata (key generation kg,
// PUT-pending set, SIEVE, allocator), mailboxes.
#pragma once
// RC_MONO: the monotonic-reads option (Config::mono) is compiled in only with -DRC_MONO=1 (RC_MONO=1 bash proto/build.sh
// -> build-$(hostname)-mono1/). The default 0 compiles its code out of the GET and PUT paths entirely (the measured
// build) and refuses Config::mono; P6 checking (sim.h, --mono 0) works in both.
#ifndef RC_MONO
#define RC_MONO 0
#endif
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rc {

constexpr size_t LINE = 64, PAY = 60;
constexpr size_t HDR = 16;          // object header at the start of line 0's payload: key u64, vlen u32, rsv u32
constexpr uint32_t MAX_VLEN = 8000;
constexpr int CHARGE_LAYOUT = 0, CHARGE_K11B = 1;   // Config::charge_model
constexpr size_t MAX_OUT = 8192;    // client output buffer size

inline uint32_t obj_lines(uint32_t vlen) { return (uint32_t)((HDR + vlen + PAY - 1) / PAY); }

// Protocol rules; every one can be switched off for the fixed-point tests (and to try alternatives). Three freshness
// (update) mechanisms, selected with set_update() / --update:
//   log    (default): every PUT appends a record (the key) to a ring in the owner's shared region after unpublishing
//          the AI entry; every host runs one polling thread that, every P/2, flushes + reads each owner's tail, deletes
//          the local-index entries named by new records and publishes a heartbeat = its TSC just BEFORE reading the
//          tails. A hit reads the heartbeat before the lookup and is valid only if now - heartbeat <= P after the loads.
//          A PUT returns W >= P after its cut (unpublish + append). Local-index entries never expire; R2 (safe
//          generation reuse across a wrap) is kept by restamping + sweeping (rules below).
//   lease  local-index entries expire.
//          Every host has an epoch counter advanced every T/2 (Config::lease_ns = T) by its own thread; an entry
//          records e0 = the epoch read BEFORE its slow path read the AI entry; a hit is valid only while
//          epoch - e0 <= 1 (so an entry lives between T/2 and T). No messages, no acks.
//   invalidate (= revoke without lease; kept for comparison and regression): a PUT revokes the key on every host and
//          waits for the acks; tombstones (+ slow_deadline) stop slow paths in flight from adopting the old entry.
// lease and revoke may be on at once; log excludes both. The rules tomb / slow_deadline only matter with revoke.
struct Rules {
    bool log = true;            // owner log + host polling (default); see the log_* rules below
    bool lease = false;         // local-index entries expire after epoch - e0 > 1
    bool revoke = false;        // PUT sends revokes and waits for acks; revocation leaves tombstones
    bool lease_from_ai_read = true;   // lease: e0 is read BEFORE the slow path reads the AI entry (ablation: false = at
                                      // install time; test L1-atinstall reproduces the counterexample)
    bool lease_bracket = true;  // lease: a GET reads the epoch first and last; a hit that spanned >= 2 epochs
                                // (a long pause: lines from two generations, or an old AI entry) is retried
    bool lease_tsc = true;      // lease: the slow path checks with the TSC that the epoch thread is ticking;
                                // if it is behind by more than T, the slow path advances the epoch by 2 (all
                                // entries of this host expire)
    // OFF by default (the design keeps the clock off the hit path): every GET also reads the TSC and, if the
    // epoch is overdue (no tick for longer than the tick period), advances it itself. Makes the bound hold when
    // the epoch thread is scheduled late (e.g. stress at nice 19 on an oversubscribed host).
    bool lease_clock_hit = false;
    bool put_wait = true;       // lease: a PUT returns only T after the new generation is published (or the key
                                // unpublished), so no reader still holds an entry that predates it
    bool put_wait_from_cut = true;   // lease: the T wait counts from the cut (the moment the old AI entry is removed:
                                     // unpublish at the PUT's start, else the publish); false = T after the PUT's last
                                     // step (stronger, longer PUTs)
    bool put_publish = true;    // PUT publishes the written value as a new generation (else it only unpublishes)
    bool restart_wait = true;   // lease: an owner (re)starting waits T before its first write (fill or PUT), so
                                // generations reused after a restart cannot meet a reader's live entry
    bool tomb = true;      // revocation leaves a tombstone; slow path adopts only entries with kg >= tombstone
    bool bump2 = true;     // PUT bumps kg again right after the backend write
    bool pending = true;   // owner drops fills for a key while a PUT on it is in progress
    bool filltok = true;   // fills are admitted only if their token equals the current kg
    bool flush_before_install = true;   // slow path: flush the object's lines before installing the LI entry
    bool slow_deadline = true;   // slow path older than slow_limit_ns at install time is abandoned and restarted
    // The owner removes the key's AI entry when a PUT starts. Without it, tombstones aged from the revoke are unsafe if
    // the PUT takes longer than slow_limit_ns before publishing (a reader can still read the old entry after its
    // tombstone expired). Ablation (false): test "slowput" reproduces the counterexample.
    bool unpublish_on_put = true;
    // The swap write order (true). The PUT neither bumps kg nor unpublishes nor appends a record at its start (only
    // `pending`: fills of the key are dropped while it runs), so the old value stays readable during the backend write;
    // after it, under the owner's lock: kg+1 (fills taken before are dropped), the new object replaces the old one in
    // place (the AI entry is switched), then the delete record is appended -- the cut. W (put_wait) counts from this
    // cut. Log scheme and SHARD only.
    //   XTRA / Beluga / MEGALON-style and SHARD: their own write order (put_wait = false for them: no W).
    //   Krater: put_wait = false is the committed ack; put_wait = true is RAW by waiting W after the swap.
    //   Default (false): the window order (cut at the PUT's start, see DELETE below), Krater's RAW by the window.
    bool put_cut_at_publish = false;
    // The swap's records. true (default): delete record (the cut), then the refresh record of the new object: a reader
    // host that holds the key ends with its entry repointed to the new object. false: refresh then delete (test variant
    // feed-swap-bird): the entry ends marked deleted, one extra slow path per holding host and PUT. Both orders are
    // model-checked (G1-G4).
    bool swap_refresh_after_del = true;
    // Ablation: the delete record (cut) BEFORE the switch to the new object; a slow path between them reads the old AI
    // entry after the cut and installs it (breaks G4). Test C2-swaporder reproduces the counterexample.
    bool swap_del_first = false;
    // Ablations of Config::xtra_own_freshness (test suite xtra; Coordination Directory versions): the owner moves the
    // key's slot version inside a PUT (its publish and after the switch) / at an eviction. xtra_kill_on_put off: an entry
    // for the replaced object keeps passing (test G1-putreturn must fail). xtra_kill_on_evict off: safe with key slots
    // (the next PUT on the key moves the slot anyway; test variant xtra-own-nokillevict, informational), kept as XTRA's rule.
    bool xtra_kill_on_put = true;
    bool xtra_kill_on_evict = true;
    // PRIV: the writer installs its own value with the fill token captured when its PUT completed (not read at
    // install time, when a later PUT may already have finished). Test "privwriter".
    bool priv_writer_token = true;
    // ---- log; each rule's counterexample is a fixed-point test in tests.cc (suite "log"), failing with the rule off
    bool put_serial = true;          // M5: the owner runs the PUTs of a key one at a time (the next PUT's cut comes after
                                     // the previous one published; W waits overlap; see exec_lanes). Ablation (false): two
                                     // executor threads per owner, PUTs of one key may interleave (log scheme only);
                                     // test M5-serial
    bool log_ts_before_tails = true; // M4: the poller takes its heartbeat timestamp BEFORE it flushes / reads the first
                                     // tail (false = after the tail reads: a cut between them is covered by a heartbeat
                                     // that claims it; test M4-tsbefore)
    bool log_wait_from_cut = true;   // M4: the PUT's W counts from its cut (record + tail written back, AI entry gone);
                                     // false = from the PUT's start, before the append (test M4-waitfromcut)
    bool log_hb_before = true;       // the hit reads the heartbeat BEFORE the local lookup (ablation: false = after
                                     // the loads; test G2-hbafter)
    bool log_seqlock = false;        // alternative, also model-checked safe (test variant log-seqlock): heartbeat read
                                     // before and after, both must be equal (in addition to the age check)
    bool log_hit_clock = true;       // the hit checks now - heartbeat <= P after its loads (ablation: without
                                     // it a stalled poller serves stale / torn hits; tests G6-pollstall, W4-midstall)
    bool log_install_check = true;   // the slow path installs only if no record for the key was applied (and no
                                     // drop-all happened) since s0 (ablation; test G3-installchk)
    bool log_snap_before = true;     // s0 is taken BEFORE the slow path reads the AI entry (ablation; test G3-installchk)
    bool log_unpublish_first = true; // the PUT unpublishes the AI entry, then appends (ablation; test G4-appendfirst)
    bool log_every_put = true;       // every PUT appends, published or not (ablation: false = only a PUT that publishes
                                     // a new generation appends, after publishing; test G5-atpublish)
    bool log_sweep = true;           // R2: each poll drops entries whose generation is >= half a round old
                                     // (ablation; test W1-sweep)
    bool log_dropall_lag = true;     // R2: if the owner's counter moved >= half a round since the last examination,
                                     // drop all its entries (ablation; test W2-dropall)
    bool log_overrun = true;         // a poll that fell behind by more than the ring (tail - applied > capacity,
                                     // or a record line of a later lap) drops all the owner's entries (test G7-overrun)
    bool log_epoch = true;           // a new log epoch in the tail line (new owner / restart) drops all entries
                                     // (test G8-epoch)
    bool restamp = true;             // R2: the owner rewrites a live object with a new generation before it is a quarter
                                     // round old (all update mechanisms; test W3-restamp)
    // ---- change feed and kg read from CXL; suite "feed" in tests.cc
    bool feed = true;            // after every publish (fill, PUT's new version, restamp) the owner appends a refresh
                                 // record (key, slot, generation); pollers repoint the entries they hold (false = the
                                 // lazy log: delete records only)
    bool feed_del = true;        // a PUT also appends its delete record at the cut (ablation; test F1-nodel)
    bool refresh_flush = true;   // a poller flushes the object's lines before repointing an entry (ablation: a torn
                                 // object after a generation wrap; test F4-refreshflush)
    bool feed_restamp = true;    // a restamp appends a refresh record too (optional: the model is safe without it)
    bool kg_cxl = true;          // the fill token is the owner's kg bucket word in CXL, loaded by the reader; no token
                                 // request message (false = token request over the control channel)
    bool kg_early = true;        // the token is loaded BEFORE the backend read (ablation: a P3 error; test F3-kglate)
    bool keep_failed = true;     // feed (default on): an entry whose fast-path check failed (slot reused, object evicted)
                                 // stays held, marked invalid, so a later refresh record (re-admission) repoints it
                                 // (refresh records apply to held entries, valid or invalidated; test F5-keepfailed).
                                 // false = erase it (then it is no longer held: no refresh)
    // ---- DELETE. A DELETE (Client::del) takes its PUT's steps and ack level without publishing anything: window order
    // kg+1, unpublish the AI entry, the delete record (= the cut), the backend tombstone, bump2, reply W after the cut;
    // swap order (and XTRA's own freshness) the backend tombstone, then under the owner's lock kg+1, the old object removed
    // (AI := None; XTRA: the key's slot version moves), the delete record (= the cut), committed reply or RAW after W.
    // EXPIRE (Client::expire, the replay's TTL) is the same without the backend write. Ablations (default on; PUTs are
    // never affected); tests D1-delete / D2-stalefill reproduce the counterexamples:
    bool del_unpublish = true;   // false (P3): the deleted key's object stays cached and published (test D1-delete)
    bool del_bump = true;        // false (P3): no kg bump in a DELETE; a fill read before it is admitted after (D2-stalefill)
    bool del_record = true;      // false: no delete record (invalidate: no revokes; PRIV: no round 1); stale reads after
                                 // the delete (test D1-delete)
    // Config::mono ablation: false = a table entry expires at once, i.e. "at any time" (expected to violate P6; test
    // M1-ghost reproduces the counterexample)
    bool mono_hold = true;
};
// --update log|lease|invalidate: sets log / lease / revoke (other rules untouched). Returns false for an unknown name.
bool set_update(Rules &r, const std::string &name);
// SHARD's measured round trips: "<dir>:<kind>:<client>" names the tables <dir>/<kind>_<client>_<vs>.txt read by
// bench/rpc/rpc_dist.h (e.g. "dist:rdma:all"). nullptr on error.
std::shared_ptr<const struct RpcDist> load_rpc_dist(const std::string &spec);
const char *update_name(const Rules &r);
// --obj-read copy|prefetch|mstore|both (= prefetch+mstore) -> Config::obj_read_mode (0..3); -1 for an unknown name
int obj_read_mode_parse(const std::string &s);
const char *obj_read_mode_name(int m);

struct Stats;

// The compared designs, all in the same framework, replayer and checker:
//   RACKCACHE    the design, Krater (default)
//   CXL_UC       Beluga-style: RackCache, but every fast-path read flushes the object's lines first (no CPU-cache reuse)
//   VER          no local index: every GET flushes + reads the AI entry, then reads the object
//   VER_NOFLUSH  as VER without flushing the AI entry (lower bound)
//   GSTATE_CXL   MEGALON-style: every GET first flushes + reads a shared state word of the owner (zero coherent
//                region), then takes the RackCache path
//   GSTATE_COH   as GSTATE_CXL but the state word is read without flushing (coherent region, emulated SCR)
//   PRIV         every host caches privately in its own DRAM; a PUT invalidates every host (two rounds through the
//                key's home host, with the same fill-token / pending / second-bump rules as RackCache)
//   XTRA         XTRA-style, after XTRA §4.1's Coordination Directory: a directory of xtra_dir_slots 8 B words
//                (default 2^20, ~8 MB, split evenly over the owners), indexed by a hash of the object's slab offset; the
//                owner stores the object's generation into its slot (+ clwb) whenever it publishes it, and every GET
//                flushes + loads the slot of the object it is about to read (fast and slow path), then takes the
//                RackCache path. Unlike GSTATE_CXL, no single state word per owner is flushed by every core.
//                Config::xtra_own_freshness: XTRA's own freshness check instead of Krater's log.
//   SHARD        keys are sharded by owner; each owner caches only its own keys in its own DRAM; a non-owner GET is
//                an RPC to the owner (request/reply over the control channel, padded to Config::rpc_ns: injected
//                delay); on a miss the requester reads the backend and sends the value to the owner (look-aside, fill
//                token as in RackCache); a PUT runs at the owner only
enum class Scheme { RACKCACHE, CXL_UC, VER, VER_NOFLUSH, GSTATE_CXL, GSTATE_COH, PRIV, SHARD, XTRA };
const char *scheme_name(Scheme s);
bool scheme_parse(const std::string &name, Scheme *out);
uint64_t k11b_slab(uint64_t bytes);              // Config::charge_model = CHARGE_K11B: the shared slab class
uint64_t charge_k11b(Scheme s, uint32_t vlen);   // CHARGE_K11B: an object's pool charge for scheme s

struct Config {
    Scheme scheme = Scheme::RACKCACHE;
    int hosts = 4;
    size_t budget_bytes = 64 << 20;  // per owner: slab bytes charged to resident objects (size-class rounded)
    size_t slab_bytes = 128 << 20;   // per owner: slab region (>= budget; absorbs size-class fragmentation)
    size_t ai_slots = 1 << 20;       // per owner, power of two
    size_t ring_lines = 1 << 14;     // per (sender, receiver) ring, power of two
    int heat_batch = 16;             // B: a client flushes non-owner hit reports every B requests (GETs and PUTs);
                                     // 0 = non-owner hits are never reported (SIEVE sees only the owner's own accesses)
    // A non-owner fast hit is reported at most once per window per local entry (the entry remembers the window it was
    // last reported in). Window = this host's poll rounds / heat_window_polls (log), else TSC time / heat_window_ns.
    // heat_dedup = false: every non-owner hit is reported
    bool heat_dedup = true;    // on by default
    uint32_t heat_window_polls = 1;
    uint64_t heat_window_ns = 1'000'000;
    int max_probe = 64;              // AI probe limit for readers (beyond it: treat as absent = miss)
    int shm_node = -1;               // NUMA node of the shared area (-1: no binding)
    // One process per host: the shared area is the POSIX shm object shm_name, created beforehand with
    // Cache::shm_create(); this process runs only host local_host and never touches other hosts' local state.
    // Threads mode: shm_name empty (anonymous shared mapping) and local_host = -1 (all hosts in this process).
    std::string shm_name;
    int local_host = -1;
    Stats *shared_stats = nullptr;   // optional: per-host Stats array (hosts entries) living in shared memory
    // Slow-path time limit T: a slow path that took longer than T is abandoned and restarted;
    // a tombstone older than T is reclaimed.
    uint64_t slow_limit_ns = 10'000'000;
    uint64_t lease_ns = 1'000'000;   // lease T; the epoch advances every lease_tick_ns (default T/2)
    // Epoch period. With T/2 an entry lives up to exactly T, the PUT waits exactly T: zero slack, so a ticker that
    // wakes up late breaks the bound. A shorter period (e.g. T/3: entries live <= 2T/3)
    // leaves the rest of T as slack for late ticks. 0 = T/2.
    uint64_t lease_tick_ns = 0;
    // A PUT whose owner does not answer within this time fails (Client::put returns 0); the backend is never
    // written by anyone but the owner. The PUT may still be executed later if the owner comes back.
    uint64_t put_timeout_ns = 2'000'000'000;
    // log: P = how old a heartbeat may be at a hit (staleness bound); the poller polls every log_interval_ns (0 = P/2);
    // a PUT returns log_wait_ns = W after its cut (0 = P; the wait is measured with the same TSC as the hit check, so
    // no clock-rate margin is needed within one machine)
    uint64_t log_poll_ns = 1'000'000, log_interval_ns = 0, log_wait_ns = 0;
    size_t log_lines = 8192;         // log ring per owner, in 64 B lines: lap header + 3 records of 16 B (key, slot, gen)
    // R2 sweep: local-index slots the poller examines per poll; 0 = the whole table every poll. The sweep may be partial
    // (model-checked) as long as, during one pass over the whole table, the owner counter's advance plus the sweep
    // threshold (half a round) stays below a full round:
    //     log_max_alloc_per_s x T_pass + 2^(gen_bits-1) < 2^gen_bits,  T_pass = ceil(li_slots / slots) x interval
    // Defaults: 2^20 slots / 16 per poll x 0.5 ms = 32.8 s per pass; at 1.22 M allocations / s per owner half a round
    // (2^31) takes 29.3 min: margin 53.7x. The cache refuses to start with a margin < 1 and warns below 10 (startup
    // check; log_sweep_margin in the config line). A small partial sweep per poll keeps the idle poller's CPU low.
    size_t log_sweep_slots = 16;
    // upper bound on one owner's allocations per second, for the sweep-margin check above; 0 = no check (tests that drive
    // allocations step by step with tiny gen_bits)
    double log_max_alloc_per_s = 1.22e6;
    // A Bloom filter in front of the local index (log_bloom_bits bits per local-index slot, 3 probes; 0 = off;
    // log scheme only). A log record the filter rules out skips the stripe lock and the lookup; the per-key-bucket record
    // sequence (install check) is updated for every record first. Why it is safe (model-checked): an item is
    // added in the same step as its install, under poll_mu (LiIndex::put, before the table install; under the log scheme
    // every put holds poll_mu); deletions never remove it (false positives only); a rebuild (once the adds since the last
    // one exceed li_slots) runs between two poll rounds, holding poll_mu. The item is what the table stores: the key
    // (32 B entries) or tag(key, owner) (li16), so a rebuild regenerates the filter from the table
    uint32_t log_bloom_bits = 8;
    // A tuning switch, off by default: a refresh record for a held entry whose ref bit is clear (not read since it was
    // loaded, or since the CLOCK hand last passed) only marks the entry deleted -- no object flush, no repoint; its next
    // read takes the slow path
    bool log_refresh_hot_only = false;
    // A poll interval below log_sweep_ref_ns sweeps
    // log_sweep_slots x interval / log_sweep_ref_ns slots (>= 16), i.e. the same slots per second as at the reference
    // interval (default 0.5 ms = the default P / 2); 0 = always log_sweep_slots per poll
    uint64_t log_sweep_ref_ns = 500'000;
    // A poll round flushes every owner's tail line, then one mfence, then reads them all (the N remote misses
    // overlap); false = one flush + mfence + load per owner. Either way the heartbeat is taken before the first flush
    bool log_batch_tails = true;
    // A poll reads up to log_prefetch_lines record lines of an owner at once: flush them all, one mfence, then load
    // them (each line is still flushed before it is read, as one line at a time; the remote misses overlap). 1 = one
    // line per flush + mfence + load
    uint32_t log_prefetch_lines = 8;
    // Compact record lines (rackcache.cc, owner log): an 8 B line header, then 7 slots of 8 B; a delete record is one
    // slot (the key), a refresh record two (key, then slot | class | generation). false = 3 records of 16 B per line
    bool log_rec8 = true;
    size_t kg_buckets = 1 << 16;     // kg words per owner in CXL (power of two; keys sharing a word only refuse more fills)
    // A host holds at most li_cap_factor x (objects cached by all owners) local entries (CLOCK on reads; refreshes do
    // not count as reads); entries pushed out take the slow path when read next. 0 = unlimited. At least li_cap_min.
    double li_cap_factor = 1.0;
    // true (default): the CLOCK queue keeps an "in the queue" mark per entry (li16: per slot index; 32 B entries: per
    // key), so an install of an entry already queued does not push it again and the queue never holds duplicates; the
    // dedup pass (O(queue) under poll_mu, once the queue passes 2 x entries + 4096) is then not needed and is skipped.
    // XTRA-style own freshness erases and reinstalls an entry on every slot-version change, which without the mark
    // pushes a duplicate each time. false = no mark: duplicates are removed by the dedup pass.
    bool li_clock_inq = true;
    uint64_t li_cap_min = 64;
    int gen_bits = 32;               // generations are the owner's allocation count mod 2^gen_bits (0 skipped); tests
                                     // use small values to reach a wrap; restamp at 2^(b-2), sweep at 2^(b-1)
    size_t li_slots = 1 << 20;       // local index capacity per host (power of two; lock-free readers)
    // Memory budget: budget_bytes covers objects (size-class rounded, including per-line gen and the header), the
    // owner's AI table (fixed: ai_slots x 16 B), one local-index entry per resident object on every host
    // (hosts x li_entry_bytes), and tombstones (every PUT leaves one per host for slow_limit_ns: hosts x ts_entry_bytes).
    // This is the CHARGE_LAYOUT accounting; charge_model describes the default. Entry sizes are those of a compact
    // implementation (the prototype's own maps are larger; they are a correctness vehicle).
    uint32_t li_entry_bytes = 16, ts_entry_bytes = 16;   // 16 B LI entry (slot + generation + fingerprint), as in li16
    // Validation only (comparing hit rates with a hit-rate simulator): charge each object its value length and
    // nothing else (no size-class rounding, AI table, LI entries or tombstones), as such a simulator does.
    bool charge_value_bytes_only = false;
    // A PUT to a cached key counts as an access (as in hit-rate simulators): the new object takes the old one's SIEVE
    // position with the visited bit set. Otherwise it is inserted at the head like a new object.
    bool write_is_access = true;
    // SHARD: a non-owner GET takes at least this long (measured RPC round trip; the control-channel round trip
    // inside it is real, the rest is busy-waited). 0 = no padding.
    uint64_t rpc_ns = 0;
    // SHARD: if set, a non-owner GET is padded to a round trip sampled from this measured distribution (by the
    // reply's size) instead of rpc_ns. Tables in the bench/rpc/rpc_dist.h format, loaded with load_rpc_dist().
    std::shared_ptr<const struct RpcDist> rpc_dist;
    // A non-owner GET reads the owner's private cache structure directly (same lock-free path as the owner; this
    // machine is coherent), then is padded to the target round trip (rpc_ns / rpc_dist); on a miss the fill token is read
    // directly too; fills and PUTs still go to the owner over the control channel. The owner's CPU for serving the GET is
    // not spent (favours the baseline). Threads mode only (each host's private cache lives in its process).
    bool shard_direct = false;
    size_t xtra_dir_slots = 1 << 20;   // XTRA directory words in total (8 B each), split over the owners (power of two each)
    // true: XTRA-style decides freshness on its own; no Krater log records, no heartbeat check, no log poller (only a
    // housekeeping round every xtra_hk_ns for the local-index cap and the R2 sweep). XTRA's Coordination Directory
    // semantics (§4.1 / §4.3): each directory word is the VERSION COUNTER of a slot chosen by key; the owner moves it
    // whenever an object of the slot is published, replaced by a PUT or evicted; a local entry keeps the version its slow
    // path read BEFORE the AI entry (XTRA's shadow version, in the entry's 32-bit generation field; the object's generation
    // is then taken from its line 0); a hit needs slot == entry's version, else the slow path reinstalls with the new
    // version (keys sharing a slot revalidate once per change). PUTs use the swap order and reply at once (RAW without W).
    // false (default) = the "borrowed" XTRA-style (Krater's log, poller and heartbeat; the offset-hashed word holds the
    // last published generation and is only counted), kept to measure the bias.
    bool xtra_own_freshness = false;
    // XREP (XTRA-style own freshness only, as xtra_own(): the log, no lease / revoke): one copy of each owner's directory
    // per reader host, in the owner's region (replica h right after replica h - 1; replica 0 is the single directory's
    // place). The owner writes every copy where it wrote the single word (all before the caller's fence); host h reads
    // only copy h.
    // 0 (default): a single directory; 1: a copy per host, written with store + clwb (owner can retain clean lines);
    // 2: a copy per host, store + clflushopt on ALL copies, including 0, to evict the owner's residual lines.
    // In mode 2 the next bump still loads its counter from copy 0, paying the extra memory read.
    // Like the directory itself, the copies are not charged to the budget.
    uint8_t xtra_dir_replicas = 0;
    uint64_t xtra_hk_ns = 10'000'000;   // XTRA own freshness: housekeeping interval
    bool li16 = true;    // local index with 16 B entries (fingerprint + owner, no key; see LiTable16 in rackcache.cc).
                         // On by default; takes effect with the log on a shared-area scheme only (lease / invalidate /
                         // PRIV / SHARD keep the 32 B layout)
    // Diagnostics: TSC timing of the log slow path (total, and the wait for poll_mu before the install)
    // and of each poll round's poll_mu hold; log2 histograms in Stats (diag_*). Off by default.
    bool diag_lock_timing = false;
    bool diag_put_timing = false;   // PUT segment timestamps (Stats::diag_put_*; replay --diag-put)
    // The executor's timer slack (ns; 0 = leave Linux's default 50 us, with which its done_q wait_until for a PUT's W
    // wakes up late by about that much). Default 1 ns, as the poller's.
    uint64_t exec_timer_slack_ns = 1;
    // PUT executor threads per owner; a key's lane is its kg bucket mod exec_lanes, so PUTs of one key stay serial
    // (M5 is per key) and every kg word keeps one writer. Every scheme (log, lease, invalidate, priv, shard; the ack
    // state of invalidate / priv is per lane); put_serial off (M5 ablation) uses one lane.
    // 1 = one executor per owner. At most 255.
    int exec_lanes = 4;
    // A lane's executor does not block on the backend write. Steps 1-2 of a PUT run, the write is
    // issued (Backend::put_latency_ns), the lane takes the next PUT; when the write completes the same lane applies it and
    // runs step 3 (bump2, publish) and replies at max(due, publish). PUTs of one key stay strictly serial (the next starts
    // after the previous replied); every kg write stays on the bucket's lane. Log update only (not lease / invalidate /
    // priv / shard, and not with put_serial off); a backend without put_latency_ns falls back to the blocking path.
    bool put_async = true;
    uint64_t put_spin_ns = 30000;   // put_async: the lane spins (instead of sleeping) this long before its next event
    // How an idle async lane waits for its next PUT: 0 = condition variable (the handler wakes it), 1 = spin,
    // > 1 = poll, sleeping this many ns between checks (timer slack exec_timer_slack_ns)
    uint64_t put_idle_ns = 0;
    // When a key's next PUT may start (step 1) in the async executor. false (default): after the previous PUT of the
    // key replied. true: after it published (what M5 requires and what the blocking executor does: "the next PUT's
    // cut comes after the previous one published; W waits overlap"), so a key's PUTs overlap only in their W waits.
    bool put_key_after_publish = false;
    // The handler polls its rings and, after 512 empty rounds, sleeps this long (0 = never sleeps: one busy core
    // per host); it takes the executor's timer slack (exec_timer_slack_ns), so a 20 us sleep is ~20 us, not ~70 us
    uint32_t handler_idle_us = 20;
    // What an object is charged against the budget. CHARGE_K11B (default): per-design object formats (K11b) under one
    // memcached-style slab allocator (64 B, then x 1.25 rounded up to 64 B), whatever the physical layout: Krater
    // (rackcache, ver*) slab(v + 4 B per 60 B) + 16 B AI entry; xtra slab(v) + 24; cxl_uc (Beluga-style) slab(v + 16)
    // + 24; gstate_* (MEGALON-style, pool only) slab(v); shard / priv slab(v) + 16. The AI table is then not charged as
    // a whole (its 16 B per object are in the formula). Host-local entries as with CHARGE_LAYOUT (li_entry_bytes per
    // host: rackcache / ver* / gstate_*). CHARGE_LAYOUT: size-class lines x 64 B (+ the AI table) for the shared area,
    // value + 16 B header for priv / shard.
    int charge_model = 1;
    // Hit check clock: the hit's "now" must not be read before its line loads complete. 0 = lfence; rdtsc; lfence;
    // 1 (default) = rdtscp alone (waits for earlier loads, does not hold later instructions; cheaper per hit and
    // sufficient for the check). Only the hit check uses it; the poller's heartbeat stamp and the PUT's cut / W spin
    // keep lfence; rdtsc; lfence.
    int hit_clock = 1;
    // obj_read: 0 = copy (per-line 60 B copies), 1 = prefetch the object's lines first (objects of
    // >= obj_prefetch_min_lines lines), 2 = masked-store copies, 3 = both (prefetch+mstore, the default). Same loads and
    // checks in every mode. Masked stores need AVX-512 BW (compile time and CPU): without it the mstore bit is dropped
    // (stderr, config line).
#ifndef RC_OBJ_READ_MODE_DEFAULT
#define RC_OBJ_READ_MODE_DEFAULT 3   // prefetch+mstore
#endif
    int obj_read_mode = RC_OBJ_READ_MODE_DEFAULT;
    uint32_t obj_prefetch_min_lines = 2;   // 2: every object with a line to prefetch (no size threshold)
    int li16_fp_bits = 20;   // fingerprint bits (20 in the 16 B layout; tests / stress use 1-4 to force aliasing)
    int li16_ep_bits = 16;   // drop-generation bits per entry (16; tests / stress use 1-2 to exercise the wrap clear)
    bool bind_hosts = false;         // pin host h's handler/executor threads to NUMA node h % nodes
    // Keys are owned by hosts 0 .. owners-1 only (0 = every host owns keys); the other hosts are pure readers
    int owners = 0;
    // A miss waits at most tok_timeout_ns for its owner's fill token; then it reads the backend and does not fill, and
    // this client sends no token request to that owner for tok_backoff_ns (owner unreachable: its keys stay readable)
    uint64_t tok_timeout_ns = 50'000'000, tok_backoff_ns = 100'000'000;
    // after a PUT timed out (put_timeout_ns), this client's PUTs to that owner fail at once for put_backoff_ns
    uint64_t put_backoff_ns = 1'000'000'000;
    // Baseline "delayed reuse" (PCCIndex-style): an evicted or replaced object's slot (and its budget) is reused only
    // after every other live host has dropped its local entry and flushed the lines and acked (M_RECL / M_RACK).
    // reuse_timeout_ns > 0: hosts that have not acked within it are declared dead (no fencing) and never waited for again
    bool reuse_delayed = false;
    uint64_t reuse_timeout_ns = 0;
    // Invalidation (revoke) PUTs wait at most this long for acks; hosts that did not ack are declared dead and never
    // revoked or waited for again (no fencing: a host that wakes up still holds entries it was never told to drop).
    // 0 = wait forever (default)
    uint64_t revoke_timeout_ns = 0;
    // Test only: emulate non-coherent reader caches in software. Every object-line load of host h goes through
    // a per-host line cache (filled from memory on a miss, never updated by the owner's writes); a flush drops
    // the lines from it; Cache::emu_prefetch() plays the hardware prefetcher.
    bool emu_noncoherent = false;
    // Read-only peer blocks: with one process per host (shm_name + local_host), map every other owner's region read-only
    // -- slab, AI table, outgoing rings, consumer tails, GSTATE word, log, kg words, XTRA directory: the whole per-owner
    // block (page-aligned), mprotect(PROT_READ) right after the mmap; only the local host's own block stays writable.
    // This enforces that only the owner writes its region. No reader writes into another host's block (its ring consumer
    // counters are in its own block; heat, fills and token requests are messages on its own outgoing rings; the heartbeat
    // is host-local), so the protocol is unchanged; a stray store there is a SIGSEGV. Readers still flush (clflushopt)
    // peer lines: permitted on a read-only page (tests --suite ro checks load / clflush / clflushopt / clwb and the
    // faulting store). Default off (the mapping is read-write); needs one process per host (threads mode: refused).
    bool ro_peers = false;
    // Monotonic reads (P6) on each host with a small expiring table, kg floors (no version in the object). Neither
    // Krater nor the baselines give P6 without it: e.g. a key evicted without a record leaves a reader's entry on its
    // ghost slot; a committed PUT writes the backend; the reader's GET misses (stale heartbeat) and returns the new value
    // from the backend; its next GET (fresh heartbeat) hits the ghost: the old value (test M1-ghost). The option (log
    // scheme, kg in CXL; not XTRA's own freshness, not VER; compiled in only with RC_MONO=1):
    //   * a GET that returns from the backend (miss, absent, the option's backend read) records (key, floor) in the host's
    //     table, floor = the key's kg loaded from CXL AFTER the data read; a slow-path return records floor = the returned
    //     object's kg - 1 (the AI entry's kg = the kg of its install: a PUT's after its bump, a fill's token); a record
    //     merges by max;
    //   * a GET reads the table ONCE at its start and keeps what it read;
    //   * a hit of a key in the table reads AI[k] (flush + load) after the object's lines and serves the copy only if AI[k] still
    //     names it (slot and generation) and its kg > floor; a slow path of a key in the table whose AI object has kg <= floor;
    //     (stored as need = floor + 1: kg >= need, so a fresh key's kg 0 needs no -1)
    //     either way the GET reads the backend instead (no install, no fill) and records a floor;
    //   * an entry lives FD + P (mono_fd_ns + log_poll_ns) after its last record: FD bounds a PUT's backend write -> cut (swap
    //     order; window order cuts before the write). The prototype has no failure detector, so the executor counts PUTs whose
    //     cut came later than FD after their backend write (Stats::mono_fd_overrun: must stay 0 for the bound to hold).
    // Table: mono_buckets buckets per host of one 64 B line (seqlock, 3 entries); a full bucket is marked overflowed for FD + P
    // instead of dropping a live entry: every key of it then takes the backend (counted). Default off: nothing changes.
    bool mono = false;
    uint64_t mono_fd_ns = 10'000'000;
    size_t mono_buckets = 1 << 14;
    Rules rules;
    // Called by the owner (lock held) every time it publishes an AI entry: (key, version). Used by the
    // checker for property P3. May be empty.
    std::function<void(uint64_t, uint32_t)> on_publish;
    // Called by the owner (lock held), log scheme, once a PUT's version and its cut are both known --
    // swap order at the cut (after the delete record); window order when it publishes, with the cut taken at its start:
    // (key, version, CLOCK_MONOTONIC ns of the cut). The tests' checker derives bounded staleness from it (a GET that
    // starts more than P after the cut, and after the publish, returns this version or newer). May be empty.
    std::function<void(uint64_t, uint32_t, uint64_t)> on_cut;
};

// Backend interface. get() returns the current version and writes its value; put() installs a new
// version, returns it and writes its value (the simulated backend generates PUT values itself).
struct Backend {
    virtual ~Backend() = default;
    virtual uint32_t get(uint64_t key, char *out, uint32_t *vlen) = 0;
    virtual uint32_t put(uint64_t key, char *out, uint32_t *vlen) = 0;
    // Asynchronous write: put_latency_ns says how long this write takes (~0 = no async support: only put());
    // the caller applies it at that time with put_complete(), which has put()'s effect without waiting.
    virtual uint64_t put_latency_ns(uint64_t key) { return ~0ULL; }
    virtual uint32_t put_complete(uint64_t key, char *out, uint32_t *vlen) { return put(key, out, vlen); }
    // DELETE: del() makes the key's next version a tombstone and returns that version (a PUT after it starts a
    // new value). get() of a deleted key returns the tombstone's version with *vlen = 0 and writes only the 16 B header
    // [key u64][ver u32][0 u32] to out ("absent"). del_complete(): the asynchronous executor's apply (put_latency_ns covers it).
    virtual uint32_t del(uint64_t key) = 0;
    virtual uint32_t del_complete(uint64_t key) { return del(key); }
    // The value of (key, ver) without a backend access: the writer of a PUT knows the value it wrote (PRIV
    // installs it in the writer's private cache).
    virtual void value_of(uint64_t key, uint32_t ver, char *out, uint32_t *vlen) = 0;
};

// Test hooks: named points between protocol steps. A hook may block (tests) or sleep (chaos mode).
enum class Hook { SlowAfterAiRead, SlowAfterInstall, MissAfterToken, MissBeforeSendFill, PutAfterBump1, PutAfterAcks,
                  PutAfterBackend, OwnerAfterGenStore, PrivWriterBeforeInstall, FastAfterLiRead,
                  FastAfterLoad,    // log: after a hit's loads, before its heartbeat check
                  PutAfterStep1,    // between the PUT's two cut steps (log: unpublish | append; lease: after unpublish)
                  PollAfterTails,   // M4: a poll round after it read the tail(s) (batched: all; else owner 0's), key 0
                  FillDone,         // diagnostics: a miss's fill was handled at its owner (admitted or dropped) or never
                                    // sent (no token / ring full); host = the host whose miss it was
                  PutSwapMid,       // swap order (put_cut_at_publish): between the switch to the new object and the delete
                                    // record (swap_del_first: between the delete record and the switch); owner lock held
                  N };
const char *hook_name(Hook h);
using HookFn = std::function<void(Hook, int host, uint64_t key)>;
void set_hook(HookFn f);  // empty function clears

enum Kind { FAST = 0, SLOW = 1, MISS = 2 };
// what an owner's executor runs for a client: a PUT, a DELETE (backend tombstone), or an EXPIRE (TTL: the cached
// copy goes, the backend keeps its value; no new version)
enum WriteOp { W_PUT = 0, W_DEL = 1, W_EXPIRE = 2 };

// shared-memory access buckets (Stats::shm / ClientStats::shm): [op][flush, CXL line reads, CXL line writes, messages, ops]
enum ShmOp { SHM_FAST = 0, SHM_SLOW, SHM_MISS, SHM_FILL, SHM_PUT, SHM_POLL, SHM_N };
struct Stats {  // per host
    std::atomic<uint64_t> frag_evictions{0}, slow_timeout{0}, ts_reclaimed{0}, get_fast{0}, get_slow{0}, get_miss{0}, fast_fail{0}, slow_fail{0}, ai_absent{0},
        tomb_refuse{0}, puts{0}, put_nocache{0}, admits{0}, fill_drop_tok{0}, fill_drop_pending{0},
        fill_drop_cached{0}, fill_drop_space{0}, evictions{0}, revokes{0}, msgs_sent{0}, msgs_recv{0},
        drop_ring_full{0}, overflowed{0}, heat_applied{0}, heat_stale{0}, priv_inval{0},
        xdir_reads{0}, xdir_other{0}, xdir_stale{0},   // XTRA: directory slots read by GETs; ... holding another
                                                       // generation (hash collision or a newer write); xdir_stale: own
                                                       // freshness, hits sent to the slow path
        // Config::mono: GETs that found their key in the table; ... that read the backend instead of a hit / slow path;
        // table records; records that met a full bucket (it went overflowed); GETs sent to the backend by an overflowed
        // bucket; owner: swap-order PUTs whose cut came more than mono_fd_ns after their backend write
        mono_table{0}, mono_backend{0}, mono_records{0}, mono_overflow{0}, mono_ovf_backend{0}, mono_fd_overrun{0},
        // ... the backend reads by reason (a hit: AI[k] no longer names the copy / the copy's kg < the need; a slow path: the AI
        // object's kg < the need; mono_ovf_backend: an overflowed bucket); table GETs that were plain misses (no copy)
        mono_bk_moved{0}, mono_bk_kg{0}, mono_bk_slow{0}, mono_table_miss{0},
        poll_cpu_ns{0},      // CPU time of this host's poller thread (CLOCK_THREAD_CPUTIME_ID, rounds + sleeps + wake-ups)
        li16_owner_clears{0},   // li16: an owner's drop generation reached a multiple of 2^16: all its entries dropped
        lease_expired{0},    // fast path found an expired entry (-> slow path)
        lease_retry{0},      // a hit spanned >= 2 epochs and was retried (lease_bracket)
        epoch_stall{0},      // slow path found the epoch thread behind by > T and advanced the epoch (lease_tsc)
        tick_late{0},        // ticks that came more than 10% of the period late
        tick_late_max_ns{0}, // largest lateness of a tick
        put_failed{0},       // PUT timed out waiting for its owner
        reply_early{0},      // put_async: a PUT reply that would have gone before its due / its publish (must stay 0)
        put_async_puts{0},   // PUTs run by the asynchronous executor
        deletes{0}, expires{0},   // owner: DELETEs / EXPIREs executed (puts counts PUTs only)
        expire_dropped{0},   // ... EXPIREs that removed a cached copy (the owner's object; PRIV: this host's copy, any owner)
        get_absent{0},       // GETs that read a deleted key's tombstone from the backend (miss, no fill)
        fill_drop_restart{0},   // fill dropped because the owner is within T of its start (restart_wait)
        handler_busy_cycles{0},   // TSC cycles the host's handler spent processing messages (owner CPU)
        // log
        log_records{0},      // owner: records appended (one per PUT)
        polls{0},            // poll rounds of this host's polling thread
        poll_cycles{0},      // TSC cycles spent in poll rounds (reader CPU for the log)
        log_applied{0},      // records applied by this host's poller
        log_deleted{0},      // local-index entries deleted by applied records
        log_refused{0},      // slow path did not install: a record for the key (or a drop-all) came after s0
        log_dropall{0},      // drop-all of one owner's entries (any reason)
        log_overrun{0},      // ... because the poller fell behind by more than the ring
        log_epoch_chg{0},    // ... because the owner's log epoch changed
        log_lag_dropall{0},  // ... because the owner's counter moved >= half a round since the last examination
        hb_stale{0},         // hit rejected: heartbeat older than P (poller late or stalled) -> slow path
        dropgen_miss{0},     // local entry invalidated by a drop-all -> slow path
        sweep_dropped{0},    // entries dropped by the R2 sweep (generation >= half a round old)
        restamps{0},         // owner: live objects rewritten with a new generation (R2)
        restamp_overflow{0}, // owner: more live objects than a quarter round of generations (only with tiny gen_bits)
        // why a GET did not take the fast path (with hb_stale, fast_fail = generation / key mismatch, dropgen_miss,
        // lease_expired); the outcome is get_slow or get_miss
        slow_noentry{0},     // no local entry (never installed, refused, swept, or erased after a failed check)
        slow_logdel{0},      // the local entry was deleted by a log record (log: the poller leaves a marker)
        slow_marked{0},      // the local entry is held but marked invalid after an earlier failed check (keep_failed)
        log_refreshed{0},    // entries repointed by refresh records (lines flushed first)
        refresh_cycles{0},   // TSC cycles spent flushing + repointing for refresh records
        log_bloom_skip{0},   // records whose key the Bloom filter ruled out (no lock, no lookup)
        bloom_rebuilds{0},   // Bloom filter rebuilds (between poll rounds, under poll_mu)
        log_refresh_cold{0}, // log_refresh_hot_only: refreshes of unread entries turned into deletes
        li_evicted{0},       // local entries pushed out by the cap (CLOCK)
        li_count{0}, li_cap{0},   // gauges: local entries held, current cap
        li_clock_q{0}, li_clock_dedups{0},   // the CLOCK queue's length (gauge), its dedup passes (li_cap)
        owner_objects{0},    // gauge: objects the owner caches (published in its tail line for the readers' caps)
        slab_bump{0}, slab_free{0},   // gauges: slab bytes ever handed out (bump pointer), bytes on the free lists
        heat_reported{0},    // non-owner hit reports put into heat batches
        shard_remote{0},     // SHARD: GET round trips to another owner
        shard_overrun{0}, shard_overrun_miss{0},    // ... whose real channel round trip already exceeded the injected target (rpc_ns / rpc_dist)
        tok_timeout{0},      // a miss got no fill token in time (or skipped an owner that timed out): backend, no fill
        // gauges / counters, owner side
        owner_used{0},       // gauge: bytes charged against the owner's budget (updated by the handler every ms)
        owner_limit{0},      // gauge: the owner's effective budget
        reclaim_pending{0},  // gauge (delayed reuse): slots waiting for acks
        reclaim_done{0}, reclaim_timeout{0},   // delayed reuse: slots freed after all acks / after declaring hosts dead
        admit_wait_reclaim{0},   // delayed reuse: admission (fill or PUT publish) refused, its space is still being reclaimed
        revoke_dead_declared{0},   // hosts declared dead by an invalidation PUT's ack timeout (revoke_timeout_ns)
        dead_hosts{0};       // gauge (delayed reuse): bit mask of hosts declared dead
    std::atomic<uint64_t> shm[SHM_N][5] = {};   // per operation (ShmOp): flush, CXL line reads, CXL line writes, messages, ops
    // diag_lock_timing: log2(TSC cycles) histograms: [0] slow-path wait for poll_mu, [1] whole log slow path (from the AI
    // read's s0 to its return), [2] poll_mu hold per poll round; diag_sum = cycle sums of the same
    std::atomic<uint64_t> diag_hist[4][48] = {}, diag_sum[4] = {};   // [3]: slow-path wait for the key's stripe lock
    // diag_put_timing: PUT segments, log2(TSC cycles) histograms: [0] client send -> owner receives, [1] owner queue
    // (received -> executor starts), [2] start -> old entry unpublished, [3] unpublished -> cut (record + tail written
    // back), [4] cut -> backend write done, [5] backend -> publish done (do_put returns), [6] publish -> reply sent (wait for
    // W incl. the done queue), [7] reply sent - due (lateness), [8] reply transit (sent -> client sees it), [9] whole PUT
    // at the client; [0..7] on the owner's Stats, [8..9] on the writer's
    std::atomic<uint64_t> diag_put_hist[10][48] = {}, diag_put_sum[10] = {};
};

class Cache;

// Hot-path counters of one client (plain integers, no shared writes on the GET path). They are added to the host's
// Stats every 1024 requests and when the client is destroyed (or by Client::flush_stats()), so Stats may lag a running
// client by up to 1024 requests. (One atomic per hit on the host's shared Stats line would cap a host's multi-threaded
// GET rate.)
struct ClientStats {
    uint64_t shm[3][5] = {};   // GETs by kind (FAST / SLOW / MISS)
    uint64_t heat_reported=0, slow_noentry=0, slow_logdel=0, slow_marked=0, tok_timeout=0, shard_remote=0, shard_overrun=0, shard_overrun_miss=0, get_absent=0, get_fast=0, get_slow=0, get_miss=0, fast_fail=0, slow_fail=0, ai_absent=0, lease_expired=0, lease_retry=0, hb_stale=0, dropgen_miss=0, log_refused=0, heat_applied=0, heat_stale=0, slow_timeout=0, tomb_refuse=0, ts_reclaimed=0, put_failed=0, xdir_reads=0, xdir_other=0, xdir_stale=0,
             mono_table=0, mono_backend=0, mono_records=0, mono_overflow=0, mono_ovf_backend=0, mono_bk_moved=0, mono_bk_kg=0, mono_bk_slow=0, mono_table_miss=0;
};

// One client thread of one host. Not thread-safe; create one per thread.
class Client {
public:
    Client(Cache *c, int host);
    ~Client();
    // GET: returns vlen and copies the value to out (>= MAX_OUT bytes); *kind = FAST / SLOW / MISS. vlen 0: the key is deleted
    // (a MISS; out holds the backend tombstone's header [key][ver][0], *ver its version).
    uint32_t get(uint64_t key, char *out, Kind *kind, uint32_t *ver = nullptr);
    uint32_t put(uint64_t key);  // returns the version written; 0 = failed (owner unreachable, see put_timeout_ns)
    // DELETE through the key's owner, same path and ack level as a PUT (Rules: del_*). Returns the tombstone's
    // version, 0 = failed. Until the next PUT a GET misses and returns vlen 0 ("absent", out = [key][ver][0]); no fill.
    uint32_t del(uint64_t key);
    // EXPIRE (TTL, the replay's --ttl-side): DELETE's protocol without the backend write. Returns 1, 0 = failed.
    uint32_t expire(uint64_t key);
    void flush_heat();
    void flush_stats();   // add this client's counters to its host's Stats now
    ClientStats totals(); // everything this client counted since it was created (flushes first)
    // Diagnostics of the last GET: why it left the fast path (0 = it did not) and, for a fast or
    // slow hit, the (owner, slot offset, generation) it read
    enum Reason { R_NONE = 0, R_NOENTRY, R_LOGDEL, R_HBSTALE, R_GENMISS, R_DROPALL, R_LEASE, R_MARKED, R_XDIR, R_N };
    // R_MARKED: the entry is held but marked invalid after an earlier failed check (keep_failed)
    int last_reason() const { return last_reason_; }
    void last_hit(int *o, uint64_t *off, uint32_t *gen) const { *o = last_o_; *off = last_off_; *gen = last_gen_; }
    int host() const { return host_; }

private:
    friend class Cache;
    uint32_t get_once(uint64_t key, char *out, Kind *kind, uint32_t *ver);
    uint32_t get_body(uint64_t key, char *out, Kind *kind, uint32_t *ver);
    // Config::mono: the GET reads the backend instead of a hit / slow path (cold and out of line, so that the GET path
    // stays as without the option when it is off; inlined, it would slow every hit)
#if RC_MONO
    uint32_t mono_backend_read(int o, uint64_t key, char *out, Kind *kind, uint32_t *ver_out, int why);
#endif
    uint32_t write_op(uint64_t key, int op);   // put / del / expire
    Cache *c_;
    int host_, id_;
    std::vector<std::vector<std::pair<uint64_t, uint32_t>>> heat_;  // per owner: (off, gen)
    int nreq_ = 0;
    uint64_t rng_ = 0;   // SHARD rpc_dist sampling
    std::vector<uint64_t> tok_skip_until_;   // per owner: mono_ns until which misses do not ask it for a token
    std::vector<uint64_t> put_skip_until_;   // per owner: mono_ns until which PUTs to it fail at once (it timed out)
    ClientStats cs_, total_;
    int last_reason_ = 0, last_o_ = -1;
    uint64_t last_off_ = 0;
    uint32_t last_gen_ = 0;
    uint32_t nstat_ = 0;
};

class Cache {
public:
    Cache(const Config &cfg, Backend *be);
    // One process per host: create, size, NUMA-bind and pre-fault the shared area for cfg; returns its size in bytes.
    static size_t shm_create(const Config &cfg);
    static void shm_unlink(const std::string &name);
    ~Cache();
    void start();
    void stop();
    int owner_of(uint64_t key) const;
    static int owner_of_static(uint64_t key, int hosts);
    const Config &cfg() const { return cfg_; }
    Stats &stats(int h);
    // test / diagnostics helpers (owner side, take the owner lock)
    bool owner_evict(uint64_t key);
    bool owner_cached(uint64_t key);
    bool held_at(int h, uint64_t key);   // diagnostics: PRIV / SHARD: host h's private cache holds key; else owner_cached
    uint64_t owner_used(int o);
    size_t owner_objects(int o);
    int64_t owner_offset(uint64_t key);                  // slab offset of the key's object, -1 if not cached
    struct Budget { uint64_t limit, ai_table, objects, li, ts, nobj; };
    Budget owner_budget(int o);                          // current charges against budget_bytes
    void wait_idle();   // threads mode: return once every ring is drained, no overflow and no PUT is executing
    void wait_writable();   // return once every local owner may write (restart_wait: T after start())
    // test helpers (local hosts only)
    void pause_epoch(int h, bool paused);                // stop / resume host h's epoch thread
    void set_unreachable(int h, bool unreachable);       // host h's handler stops / resumes reading its rings
    uint32_t epoch(int h);
    void pause_poller(int h, bool paused);               // log: stop / resume host h's polling thread (returns once
                                                         // no poll round is running)
    void poll_once(int h);                               // log: run one poll round of host h now (from this thread)
    uint32_t owner_advance(int o, uint64_t n);           // owner o allocates n generations elsewhere; returns the next
    uint32_t owner_next_gen(int o);                      // next generation owner o would allocate (ignoring restamps)
    uint32_t owner_gen(uint64_t key);                    // generation of the key's object, 0 if not cached
    void owner_reset(int o);                             // test: a new owner takes over region o (drops every object
                                                         // without records, new log from 0 in a new epoch, the
                                                         // allocation counter continues, waits W before writing)
    void emu_drop_line(int h, int o, uint64_t off);      // emu_noncoherent: host h's cache drops one line
    // diagnostics: the key's AI entry as it is now (no flush: coherent DRAM only); false if absent
    bool ai_peek(uint64_t key, uint64_t *off, uint32_t *gen);
    void emu_prefetch(int h, int o, uint64_t off);       // emu_noncoherent: host h's cache loads one line now
    void emu_drop_all(int h);                            // emu_noncoherent: host h's cache drops every line
    // ro_peers test hook: one access from this process to owner o's block, in its part `where` (0 slab, 1 AI table, 2 outgoing rings,
    // 3 consumer tails, 4 log tail line, 5 kg words, 6 the block's last line): kind 0 load, 1 clflush, 2 clflushopt, 3 clwb,
    // 4 store (rewrites the byte it read). With ro_peers and o another host's block, only 4 must fault (SIGSEGV).
    void peer_access(int o, int kind, int where = 0);
    std::string describe() const;

    struct Impl;
private:
    friend class Client;
    Config cfg_;
    std::unique_ptr<Impl> p_;
};

}  // namespace rc
