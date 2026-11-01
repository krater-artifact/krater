// Trace replayer: drives the prototype with a request trace and reports the GET hit ratio, to be compared with a
// trace-driven simulator's (--expect). --trace <run.binary> reads kv-benchmark 16 B records (e.g. Twitter cluster17);
// --bin <records.bin> reads u64 records (key id 32 | op 3 | client hash top 8 | ts 21, ids index the same trace's
// objsize.bin; tools/ycsb/trace_io.h) and --routing client sends every record of a client to host hash*N >> 64;
// --skip / --limit take a window. --perf --threads T runs T client threads per host (each serves every T-th record of
// its host). The perf report adds why GETs left the fast path, PUT latency, visibility after a PUT, and the log's
// poller / record counts. --rpc-dist injects sampled RPC round trips (SHARD).
// Matches the simulators' model: every record draws its receiving host from splitmix64(seed) (z >> 62 for 4
// hosts), key -> owner is fmix64 (Cache::owner_of), capacity M = total distinct bytes x pct / 100 split evenly
// over the owners, object size = objsize.bin[key], no warm-up, hit ratio = GET hits / GETs.
// Serial mode (the default, without --perf): one request at a time and, after a miss or PUT, wait until every fill /
// message has been processed, so admission is effectively synchronous as in the simulators.
// Every GET is checked by the independent checker (values are generated from (key, version) with the trace's
// object size). "ghost hits": GETs served (fast or slow path) from an object the owner had already evicted but
// whose slot was not reused yet; the value is still correct, but the simulators count these as misses.
#include <fcntl.h>
#include <getopt.h>
#include <numa.h>
#include <signal.h>
#include <sys/wait.h>
#include <x86intrin.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "rackcache.h"
#include "sim.h"

using namespace rc;

struct Rec { uint64_t key; uint32_t vsize; uint16_t ksize, bf; };
static_assert(sizeof(Rec) == 16, "trace record");

static inline int next_host(uint64_t &s, int hosts) {   // splitmix64, as the simulators draw the host (4 hosts: z >> 62)
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return hosts == 4 ? (int)(z >> 62) : (int)(z % (uint64_t)hosts);
}

// Trace accessor: kv-benchmark 16 B records (R) or --bin u64 records (B, with a client hash)
struct Trace {
    const Rec *R = nullptr;
    const uint64_t *B = nullptr;
    bool by_client = false;
    uint64_t key(uint64_t i) const { return R ? R[i].key : (B[i] & 0xffffffffULL); }
    int op(uint64_t i) const { return R ? R[i].bf >> 12 : (int)(B[i] >> 32 & 7); }   // 2 get, 1 set, 3 incr, 4 delete
    uint64_t client(uint64_t i) const { return (B[i] >> 35 & 0xff) << 56; }   // = rec_client in tools/ycsb/trace_io.h
    uint32_t ts(uint64_t i) const { return B ? (uint32_t)(B[i] >> 43) : 0; }   // trace time, seconds (bin records only)
};
// --pace trace: record i is issued no earlier than start + (ts(i) - ts0) / k (trace seconds / k = wall seconds)
struct Pace { double k = 0; uint32_t ts0 = 0; const std::atomic<uint64_t> *start = nullptr; };
// --pace spread: trace timestamps are whole seconds, so --pace trace releases each second's requests at once
// (a burst every 1 / k s); spread places the j-th of a second's n records at second + j / n (records are time-ordered)
static std::vector<float> g_spread;
// The receiving host of record i. The random stream is drawn for every record, whatever the routing (so the stream
// does not depend on it); with --routing client the host is client_hash * N >> 64 (one host per client).
static inline int route(const Trace &T, uint64_t i, uint64_t &rng, int hosts) {
    int h = next_host(rng, hosts);
    if (T.by_client) h = (int)(((unsigned __int128)T.client(i) * (unsigned)hosts) >> 64);
    return h;
}

// ---------------------------------------------------------------- --perf mode
// Latency histogram: 1 ns buckets below 4096 ns, then 64 buckets per power of two up to 2^40 ns.
struct Hist {
    static constexpr int LIN = 4096, SUB = 64, NB = LIN + 28 * SUB;
    uint64_t b[NB] = {0}, n = 0;
    double sum = 0;
    static int idx(uint64_t ns) {
        if (ns < LIN) return (int)ns;
        int e = 63 - __builtin_clzll(ns);   // >= 12
        int i = LIN + (e - 12) * SUB + (int)((ns >> (e - 6)) & (SUB - 1));
        return std::min(i, NB - 1);
    }
    static double val(int i) {
        if (i < LIN) return i;
        int e = (i - LIN) / SUB + 12, m = (i - LIN) % SUB;
        return std::ldexp(1.0 + (m + 0.5) / SUB, e);
    }
    void add(uint64_t ns) { b[idx(ns)]++; n++; sum += ns; }
    void merge(const Hist &o) { for (int i = 0; i < NB; i++) b[i] += o.b[i]; n += o.n; sum += o.sum; }
    double q(double p) const {
        if (!n) return 0;
        uint64_t want = (uint64_t)std::ceil(p * n), c = 0;
        for (int i = 0; i < NB; i++) if ((c += b[i]) >= want) return val(i);
        return val(NB - 1);
    }
    double mean() const { return n ? sum / n : 0; }
};
// --miss-diag (threads mode): each measured miss classified when it starts (hook
// MissAfterToken), by the state of its key: COLD = the key's first GET in the trace (this host's first, for PRIV);
// DUP_SAME / DUP_CROSS = a fill for the key is still in flight (its miss started, its owner has not handled the fill yet)
// from this host / only from other hosts; IN_POOL = the cache holding the key (owner / PRIV host) has it, yet this GET
// missed; LEFT = it was admitted before and has left (evicted, or its PUT did not republish); DROPPED = earlier fills
// were all dropped; OTHER = none of these (no earlier fill, not the first GET: e.g. first touched by PUTs only)
enum { MC_COLD, MC_DUP_SAME, MC_DUP_CROSS, MC_IN_POOL, MC_LEFT, MC_DROPPED, MC_OTHER, MC_N };
static const char *MC_NAMES[MC_N] = {"cold", "dup_same_host", "dup_cross_host", "in_pool", "left", "dropped", "other"};
static bool g_miss_diag = false;
static Cache *g_dc = nullptr;
static bool g_dpriv = false, g_dshard = false;
static uint64_t g_dN = 0;
static std::unique_ptr<std::atomic<uint16_t>[]> g_inflight_host, g_inflight_key;
static std::unique_ptr<std::atomic<uint8_t>[]> g_fillstate;   // per key (PRIV: per host x key): 1 fill handled, 2 admitted
static std::vector<uint32_t> g_first_get;                      // per key (PRIV: per host x key): record of its first GET
static thread_local int tl_miss_class = -1;
static inline size_t dkey(int h, uint64_t k) { return g_dpriv ? (size_t)h * (g_dN + 1) + k : k; }
static inline bool dheld(int h, uint64_t k) { return g_dc->held_at(g_dpriv ? h : g_dc->owner_of(k), k); }
static void diag_hook(Hook hk, int h, uint64_t k) {
    if (k > g_dN) return;
    size_t hi = (size_t)h * (g_dN + 1) + k;
    if (hk == Hook::MissAfterToken) {
        int c;
        if (g_inflight_host[hi].load() > 0) c = MC_DUP_SAME;
        else if (!g_dpriv && g_inflight_key[k].load() > 0) c = MC_DUP_CROSS;
        else if (dheld(h, k)) c = MC_IN_POOL;
        else {
            uint8_t s = g_fillstate[dkey(h, k)].load();
            c = (s & 2) ? MC_LEFT : (s & 1) ? MC_DROPPED : MC_OTHER;
        }
        g_inflight_host[hi].fetch_add(1);
        g_inflight_key[k].fetch_add(1);
        tl_miss_class = c;
    } else if (hk == Hook::FillDone) {
        uint16_t v = g_inflight_host[hi].load();
        while (v && !g_inflight_host[hi].compare_exchange_weak(v, v - 1)) {}
        v = g_inflight_key[k].load();
        while (v && !g_inflight_key[k].compare_exchange_weak(v, v - 1)) {}
        g_fillstate[dkey(h, k)].fetch_or(dheld(h, k) ? 3 : 1);
    }
}
struct HostSum {   // one host's measured part (after warm-up)
    uint64_t gets = 0, fast = 0, slow = 0, miss = 0, puts = 0, put_failed = 0;
    // why a GET left the fast path: no entry / deleted by a log record / heartbeat older than P / generation or key
    // check failed / entry dropped with all of its owner's / lease expired
    uint64_t r_noentry = 0, r_logdel = 0, r_hbstale = 0, r_genmiss = 0, r_dropall = 0, r_lease = 0;
    uint64_t early_vis = 0;   // GETs that returned a version whose PUT had not returned yet (other hosts' PUTs too)
    // GETs by (reason they left the fast path) x (outcome 0 fast, 1 slow hit, 2 miss); ghost hits (fast hits whose slot
    // no longer holds the key's published object): the key is absent from AI (evicted) / published elsewhere
    uint64_t rk[Client::R_N][3] = {};
    uint64_t ghost_absent = 0, ghost_moved = 0;
    // "no local entry" GETs on a host that had read the key before (fast or slow hit), by outcome (slow hit / miss): an
    // entry whose check failed is erased (unless keep_failed marks it, with the change feed), so these are the
    // simulator's "stale" re-reads, not first reads
    uint64_t again_slow = 0, again_miss = 0;
    // "no local entry" GETs of a key this host had never read (held is tracked from the start of the run, warm-up
    // included): first reads, by outcome (slow hit = another host brought it into the pool)
    uint64_t first_slow = 0, first_miss = 0;
    uint64_t heat_reported = 0;   // hit reports put into heat batches (measured part)
    uint64_t mc[MC_N] = {};       // --miss-diag: measured misses by class
    // --put-inflight K: PUTs handed to K writer threads; time the GET client was blocked because K PUTs were already
    // queued, the deepest queue seen, and each PUT's wait in the queue (in `putq`)
    uint64_t put_blocked_ns = 0, put_queue_max = 0;
    Hist putq;
    double busy_s = 0, wall_s = 0;
    Hist fasth, slowh;        // fast-path and slow-path hit latency separately (hit = both)
    Hist all, hit, put, vis;  // vis: PUT returned -> first GET of each other host that returns that version or newer
    // --open-rate: GET latency from its Poisson arrival (queueing included) and the queueing alone (arrival -> issue);
    // measured records offered / the measured span in ns of the arrival schedule
    Hist open_get, open_q;
    uint64_t open_recs = 0, open_span_ns = 0;
    // CPU time by role over the whole run (warm-up included): the GET client threads, the async PUT pool's writer
    // threads (CLOCK_THREAD_CPUTIME_ID at their end) and, in --procs, the host's whole process (getrusage after the cache
    // stopped: clients + PUT pool + the owner's handler / executor / poller / ticker threads); 0 = not measured
    uint64_t cpu_client_ns = 0, cpu_put_ns = 0, cpu_proc_ns = 0;
    // the client threads' time inside requests (GET calls, PUT hand-offs / synchronous PUTs; whole run): their CPU time
    // without the pacing waits (a paced client spins between requests, so its thread CPU time is mostly waiting)
    uint64_t cpu_client_work_ns = 0;
    // --llc-count: user-space LLC misses (PERF_COUNT_HW_CACHE_MISSES) of the client threads inside each measured GET, by
    // outcome (fast / slow hit / miss); the whole host process over the run (inherited by every thread created after the
    // counter) and the host's GETs over the whole run
    uint64_t llc_get[3] = {0, 0, 0}, llc_n[3] = {0, 0, 0}, llc_proc = 0, gets_whole = 0;
    int llc_mode = 0;   // 0 off, 1 rdpmc, 2 read()
    // --deletes 1 / --ttl-side: measured DELETE records run; TTL expiries the replay issued (an EXPIRE before the GET) and
    // how many of those GETs then missed; the same over the whole run (warm-up included)
    uint64_t dels = 0, ttl_expired = 0, ttl_miss = 0, ttl_expired_whole = 0;
    void merge(const HostSum &o) {
        gets += o.gets; fast += o.fast; slow += o.slow; miss += o.miss; puts += o.puts; put_failed += o.put_failed;
        r_noentry += o.r_noentry; r_logdel += o.r_logdel; r_hbstale += o.r_hbstale; r_genmiss += o.r_genmiss;
        r_dropall += o.r_dropall; r_lease += o.r_lease; early_vis += o.early_vis;
        for (int a = 0; a < Client::R_N; a++) for (int b = 0; b < 3; b++) rk[a][b] += o.rk[a][b];
        ghost_absent += o.ghost_absent; ghost_moved += o.ghost_moved; again_slow += o.again_slow; again_miss += o.again_miss;
        first_slow += o.first_slow; first_miss += o.first_miss; fasth.merge(o.fasth); slowh.merge(o.slowh);
        heat_reported += o.heat_reported;
        for (int c = 0; c < MC_N; c++) mc[c] += o.mc[c];
        put_blocked_ns += o.put_blocked_ns; put_queue_max = std::max(put_queue_max, o.put_queue_max);
        all.merge(o.all); hit.merge(o.hit); put.merge(o.put); vis.merge(o.vis); putq.merge(o.putq);
        open_get.merge(o.open_get); open_q.merge(o.open_q); open_recs += o.open_recs; open_span_ns = std::max(open_span_ns, o.open_span_ns);
        cpu_client_ns += o.cpu_client_ns; cpu_put_ns += o.cpu_put_ns; cpu_proc_ns += o.cpu_proc_ns; cpu_client_work_ns += o.cpu_client_work_ns;
        for (int k = 0; k < 3; k++) { llc_get[k] += o.llc_get[k]; llc_n[k] += o.llc_n[k]; }
        llc_proc += o.llc_proc; gets_whole += o.gets_whole; llc_mode = std::max(llc_mode, o.llc_mode);
        dels += o.dels; ttl_expired += o.ttl_expired; ttl_miss += o.ttl_miss; ttl_expired_whole += o.ttl_expired_whole;
        wall_s = std::max(wall_s, o.wall_s);
    }
};
// visibility bookkeeping per key (shared by all hosts; in --procs it lives in the simulator's shm segment)
struct Vis { std::atomic<uint32_t> *ver, *seen; std::atomic<uint64_t> *ret; };
// per host: keys this host has read with a hit at least once (diagnostic; benign races between a host's threads)
static std::vector<std::vector<uint8_t>> g_held;
static double g_tsc_ghz = 0;
static double calibrate_tsc() {
    uint64_t t0 = now_ns(), c0 = __rdtsc();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    uint64_t t1 = now_ns(), c1 = __rdtsc();
    return (double)(c1 - c0) / (double)(t1 - t0);
}
// One closed-loop client thread of host h (thread tid of nthr): scans the whole trace with the same host stream as the
// serial mode and serves every nthr-th record drawn for host h. Records before `warm` are executed but not measured.
static Pace g_pace;
static void pace_wait(const Trace &T, uint64_t i) {
    uint64_t start = g_pace.start->load(std::memory_order_acquire);
    double tsec = (double)(T.ts(i) - g_pace.ts0) + (g_spread.empty() ? 0.0 : (double)g_spread[i]);
    uint64_t target = start + (uint64_t)(tsec * 1e9 / g_pace.k);
    for (uint64_t now = now_ns(); now < target; now = now_ns()) {
        if (target - now > 200'000) std::this_thread::sleep_for(std::chrono::nanoseconds(target - now - 100'000));
        else _mm_pause();
    }
}
// --put-inflight K > 0: a client does not wait for its PUTs. A write record is handed, at its trace position, to a pool
// of K writer threads (each its own Client of the same host); the client blocks only when K PUTs are already queued
// (put_blocked_ns). GET latency and the GET rate then exclude PUT waits (W); a PUT's latency is measured from the
// hand-off to its return (queueing + PUT incl. W). PUTs of the same key may overlap (as PUTs of different hosts always
// could); the checker's rules stay the same. K = 0: the PUT runs in the client (closed loop).
static int g_put_inflight = 0;
// --put-pool-away (diagnostic): host h's writer threads run on NUMA node (h + 2) mod nodes, not on h's own node
static bool g_put_pool_away = false;
// --open-rate R: open loop. Each client thread of a host draws Poisson arrivals at R / threads_per_host per second
// (R = trace records per second per host, wall clock; GET and PUT records alike, in trace order) instead of following
// the trace's timestamps, waits for each arrival and measures a GET from its arrival, so queueing behind slower requests
// is included (`open_get`; the closed-loop service time stays in `all`). Excludes --pace. Default 0 = off (closed loop).
static double g_open_rate = 0;
// --no-puts: write records of the trace are skipped (read-only run; GETs, their order and timing unchanged). Default off.
static bool g_no_puts = false;
// --lean: closed-loop throughput without the harness's per-request checks in the client loop (checker verify, ghost-hit
// diagnostic = an AI peek per fast hit, visibility tracking); the cache runs as without --lean.
static bool g_lean = false;
// --deletes 1: trace op 4 runs as a DELETE (Client::del: the PUT path's steps and ack level without a publish; the
// checker takes the tombstone's version like a PUT's, so a GET that returns the deleted value after the DELETE returned
// is an error; a GET of a deleted key misses and returns "absent", no fill). Default 0: delete records are skipped (the
// serial mode counts them in deletes_ignored).
static bool g_deletes = false;
// --ttl-side <file>: TTL expiry. One u32 per record of the --bin file (aligned with the whole file, before --skip): the
// record's effective TTL in seconds, 0 = never expires, as the simulators track it (a set-class write: the line's own ttl;
// any other record: the ttl of the key's most recent set-class write in the original text, look-aside fills included --
// so a de-backfilled window keeps its fills' TTLs). The replay keeps, per key, the trace second at which the cached copy
// expires: a write installs it (write-allocate: its second + ttl; incr / decr keep a running expiry), a GET miss refills
// it (the miss's second + ttl; deleted keys: none). A GET at second s > that expiry (whole seconds) first EXPIREs the key
// (Client::expire: the DELETE protocol without the backend write -- the cached copy goes, the backend keeps its value, no
// version), counts it, and then runs (RAW write paths: it misses and refills). One expiry per key (one shared copy;
// PRIV's EXPIRE drops every host's copy). Lazy, as in the simulators: a copy nobody reads stays until evicted. Default
// off; needs --bin; not with --procs.
static const uint32_t *g_ttl_side = nullptr;
static std::string g_ttl_side_f;
// --mono 0|1: 1 sets Config::mono (monotonic reads per host); given either way, P6 is checked and a MONO line reports it
// (0 = the control: P6 checked, Config::mono off); not given: P6 is not checked and no MONO line is printed
static int g_mono_arg = -1;
static std::unique_ptr<std::atomic<uint32_t>[]> g_ttl_exp;   // per key: expiry second of the cached copy, 0 = none
static inline uint32_t ttl_expiry_s(uint32_t now_s, uint32_t ttl) {   // now_s + ttl, saturated (0 = never)
    if (!ttl) return 0;
    uint64_t e = (uint64_t)now_s + ttl;
    return e > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)e;
}
// before GET record i: true if the key's copy had expired and this GET issued its EXPIRE (another host's GET may win the race)
template <class T>
static inline bool ttl_expire_before(Client &cl, const T &tr, uint64_t i, uint64_t key) {
    uint32_t ex = g_ttl_exp[key].load(std::memory_order_relaxed);
    if (!ex || tr.ts(i) <= ex || !g_ttl_exp[key].compare_exchange_strong(ex, 0)) return false;
    cl.expire(key);
    return true;
}
// a copy installed at record i: a write (keep: incr / decr keep a running expiry) or a GET miss's refill
template <class T>
static inline void ttl_install(const T &tr, uint64_t i, uint64_t key, bool keep) {
    uint32_t e = ttl_expiry_s(tr.ts(i), g_ttl_side[i]);
    if (!keep) { g_ttl_exp[key].store(e, std::memory_order_relaxed); return; }
    uint32_t z = 0;
    g_ttl_exp[key].compare_exchange_strong(z, e);
}
// --llc-count: count user-space LLC misses on host g_llc_host (host 1): per measured GET in its client threads (counter
// read with rdpmc through the event's mmap page when the kernel allows it, else read(); outside the timed region) and
// for the whole host process (--procs). Default off.
static bool g_llc = false;
static int g_llc_host = 1;
static int llc_open(bool inherit) {
    perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = PERF_TYPE_HARDWARE;
    a.config = PERF_COUNT_HW_CACHE_MISSES;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.inherit = inherit;
    return (int)syscall(SYS_perf_event_open, &a, 0, -1, -1, 0);
}
struct LlcCounter {   // one thread's counter
    int fd = -1;
    perf_event_mmap_page *pc = nullptr;
    bool rd = false;
    void open() {
        fd = llc_open(false);
        if (fd < 0) return;
        void *m = mmap(nullptr, (size_t)sysconf(_SC_PAGESIZE), PROT_READ, MAP_SHARED, fd, 0);
        if (m != MAP_FAILED) { pc = (perf_event_mmap_page *)m; rd = pc->cap_user_rdpmc; }
    }
    uint64_t now() {
        if (fd < 0) return 0;
        if (rd) {
            uint32_t seq;
            uint64_t v;
            do {
                seq = pc->lock;
                __asm__ __volatile__("" ::: "memory");
                uint32_t idx = pc->index;
                v = pc->offset;
                if (idx) {
                    uint64_t c = __rdpmc((int)idx - 1);
                    int w = pc->pmc_width;
                    c <<= 64 - w; c >>= 64 - w;
                    v += c;
                }
                __asm__ __volatile__("" ::: "memory");
            } while (pc->lock != seq);
            return v;
        }
        uint64_t v = 0;
        return read(fd, &v, sizeof v) == (ssize_t)sizeof v ? v : 0;
    }
    ~LlcCounter() { if (pc) munmap(pc, (size_t)sysconf(_SC_PAGESIZE)); if (fd >= 0) close(fd); }
};
static uint64_t thread_cpu_ns() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
// --backend-get-dist / --backend-put-dist: the backend's read / write latency sampled per request from a measured
// distribution (a table file: "#" comment lines, then one quantile per line in ns, line i = quantile (i + 0.5) / lines)
// instead of the fixed --delay-ns / --write-delay-ns (both set to 0 for the sampled side). Uniform index per request
// from a per-thread xorshift stream.
struct LatTable { std::vector<uint64_t> q; double mean = 0; };
static std::shared_ptr<LatTable> g_get_dist, g_put_dist;
static std::shared_ptr<LatTable> load_lat_table(const std::string &f) {
    FILE *fp = fopen(f.c_str(), "r");
    if (!fp) { perror(f.c_str()); return nullptr; }
    auto t = std::make_shared<LatTable>();
    char line[256];
    while (fgets(line, sizeof line, fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        t->q.push_back(strtoull(line, nullptr, 10));
    }
    fclose(fp);
    if (t->q.empty()) { fprintf(stderr, "%s: no quantiles\n", f.c_str()); return nullptr; }
    for (uint64_t v : t->q) t->mean += (double)v;
    t->mean /= (double)t->q.size();
    return t;
}
static inline uint64_t lat_sample(const LatTable &t) {
    static std::atomic<uint64_t> seeds{0x9E3779B97F4A7C15ULL};
    thread_local uint64_t x = seeds.fetch_add(0x632BE59BD9B4E019ULL) | 1;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return t.q[x % t.q.size()];
}
class DistBackend : public SimBackend {   // SimBackend + the sampled latencies (none set: SimBackend unchanged)
public:
    using SimBackend::SimBackend;
    uint32_t get(uint64_t key, char *out, uint32_t *vlen) override {
        if (g_get_dist) spin_ns(lat_sample(*g_get_dist));
        return SimBackend::get(key, out, vlen);
    }
    uint32_t put(uint64_t key, char *out, uint32_t *vlen) override {
        if (g_put_dist) spin_ns(lat_sample(*g_put_dist));
        return SimBackend::put(key, out, vlen);
    }
    uint64_t put_latency_ns(uint64_t key) override { return g_put_dist ? lat_sample(*g_put_dist) : SimBackend::put_latency_ns(key); }
    uint32_t del(uint64_t key) override {   // a DELETE takes a backend write's latency
        if (g_put_dist) spin_ns(lat_sample(*g_put_dist));
        return SimBackend::del(key);
    }
};
struct PutJob { uint64_t key, t_enq; bool measuring; int op = W_PUT; };   // op: W_PUT, W_DEL (--deletes 1)
class PutPool {
public:
    PutPool(Cache &cache, Checker &ck, int h, int k, const Vis &V) : ck_(ck), h_(h), k_(k), V_(V) {
        for (int i = 0; i < k; i++) th_.emplace_back([this, &cache] { worker(cache); });
    }
    // hand a PUT over; returns the ns spent blocked because k PUTs were already waiting
    uint64_t submit(uint64_t key, bool measuring, int op = W_PUT) {
        std::unique_lock<std::mutex> g(mu_);
        uint64_t b0 = 0;
        if (q_.size() >= (size_t)k_) { b0 = now_ns(); cv_space_.wait(g, [this] { return q_.size() < (size_t)k_; }); }
        q_.push_back({key, __rdtsc(), measuring, op});
        qmax_ = std::max<uint64_t>(qmax_, q_.size());
        cv_job_.notify_one();
        return b0 ? now_ns() - b0 : 0;
    }
    // wait for every queued and running PUT, stop the writers, add their measured PUTs to *out
    void finish(HostSum *out) {
        { std::lock_guard<std::mutex> g(mu_); done_ = true; }
        cv_job_.notify_all();
        for (auto &t : th_) t.join();
        out->puts += sum_.puts; out->put_failed += sum_.put_failed; out->dels += sum_.dels;
        out->put.merge(sum_.put); out->putq.merge(sum_.putq); out->cpu_put_ns += sum_.cpu_put_ns;
        out->put_queue_max = std::max(out->put_queue_max, qmax_);
    }
private:
    void worker(Cache &cache) {
        if (g_put_pool_away && numa_available() >= 0) numa_run_on_node((h_ + 2) % (numa_max_node() + 1));
        Client cl(&cache, h_);
        HostSum part;
        for (;;) {
            PutJob j;
            {
                std::unique_lock<std::mutex> g(mu_);
                cv_job_.wait(g, [this] { return done_ || !q_.empty(); });
                if (q_.empty()) break;
                j = q_.front();
                q_.pop_front();
                cv_space_.notify_one();
            }
            uint64_t c0 = __rdtsc();
            uint32_t v = j.op == W_DEL ? cl.del(j.key) : cl.put(j.key);
            uint64_t c1 = __rdtsc();
            ck_.put_done(j.key, v);
            if (v) {   // as in run_host: return time and the writer's bit first, version last
                V_.ret[j.key].store(c1, std::memory_order_relaxed);
                V_.seen[j.key].store(1u << h_, std::memory_order_relaxed);
                V_.ver[j.key].store(v, std::memory_order_release);
            }
            if (j.measuring && j.op == W_DEL) {   // a DELETE: counted, its latency is not in the PUT histograms
                if (v) part.dels++; else part.put_failed++;
            } else if (j.measuring) {
                if (v) {
                    part.puts++;
                    part.put.add((uint64_t)((c1 - j.t_enq) / g_tsc_ghz));
                    part.putq.add((uint64_t)((c0 - j.t_enq) / g_tsc_ghz));
                } else part.put_failed++;
            }
        }
        std::lock_guard<std::mutex> g(mu_);
        sum_.puts += part.puts; sum_.put_failed += part.put_failed; sum_.put.merge(part.put); sum_.putq.merge(part.putq);
        sum_.dels += part.dels;
        sum_.cpu_put_ns += thread_cpu_ns();
    }
    Checker &ck_;
    int h_, k_;
    const Vis &V_;
    std::mutex mu_;
    std::condition_variable cv_job_, cv_space_;
    std::deque<PutJob> q_;
    bool done_ = false;
    uint64_t qmax_ = 0;
    HostSum sum_;
    std::vector<std::thread> th_;
};
static void run_host(Cache &cache, Checker &ck, int h, int hosts, const Trace &T, uint64_t nrec, uint64_t warm,
                     uint64_t seed, int tid, int nthr, HostSum *out, const Vis &V) {
    if (g_pace.k > 0 || g_open_rate > 0) while (!g_pace.start->load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(100));
    Client cl(&cache, h);
    // --open-rate: this thread's Poisson arrival schedule (mean gap nthr / R), own xorshift stream
    uint64_t orng = (seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(h * 1024 + tid + 1)) | 1;
    double t_arr = g_open_rate > 0 ? (double)g_pace.start->load(std::memory_order_acquire) : 0, t_meas0 = 0;
    const double gap_ns = g_open_rate > 0 ? 1e9 * nthr / g_open_rate : 0;
    LlcCounter llc;
    const bool llc_on = g_llc && h == g_llc_host;
    if (llc_on) { llc.open(); out->llc_mode = llc.fd < 0 ? 0 : llc.rd ? 1 : 2; }
    std::unique_ptr<PutPool> pool;
    if (g_put_inflight > 0) pool.reset(new PutPool(cache, ck, h, g_put_inflight, V));
    std::vector<char> buf(MAX_OUT);
    uint64_t rng = seed, tw0 = 0, seq = 0;
    bool measuring = false;
    ClientStats base{};
    for (uint64_t i = 0; i < nrec; i++) {
        int hh = route(T, i, rng, hosts);
        if (hh != h || seq++ % nthr != (uint64_t)tid) continue;
        if (g_pace.k > 0) pace_wait(T, i);
        uint64_t t_issue = 0;
        if (g_open_rate > 0) {   // next arrival, then wait for it (a thread behind its schedule issues at once)
            orng ^= orng << 13; orng ^= orng >> 7; orng ^= orng << 17;
            t_arr += -std::log(((double)(orng >> 11) + 0.5) * 0x1.0p-53) * gap_ns;
            uint64_t target = (uint64_t)t_arr, now = now_ns();
            for (; now < target; now = now_ns()) {
                if (target - now > 200'000) std::this_thread::sleep_for(std::chrono::nanoseconds(target - now - 100'000));
                else _mm_pause();
            }
            t_issue = now;
        }
        if (!measuring && i >= warm) { measuring = true; tw0 = now_ns(); base = cl.totals(); t_meas0 = t_arr; }
        if (measuring && g_open_rate > 0) { out->open_recs++; out->open_span_ns = (uint64_t)(t_arr - t_meas0); }
        uint64_t key = T.key(i);
        int t = T.op(i);
        if (g_no_puts && (t == 1 || t == 3)) continue;
        if (t == 2) {
            const bool expired = g_ttl_side && ttl_expire_before(cl, T, i, key);   // --ttl-side: EXPIRE first
            if (expired) out->ttl_expired_whole++;
            uint32_t req = ck.begin(key);
            const uint32_t sfl = ck.stale_begin(key);   // 0 unless g_committed (bounded staleness)
            const uint32_t mfl = ck.mono_begin(h, key);   // P6 (--mono): 0 unless enabled
            Kind kind;
            out->gets_whole++;
            const uint64_t l0 = llc_on && measuring ? llc.now() : 0;
            _mm_lfence();
            uint64_t c0 = __rdtsc();
            uint32_t vlen = cl.get(key, buf.data(), &kind);
            unsigned aux;
            uint64_t c1 = __rdtscp(&aux);
            if (llc_on && measuring) { int k = kind == FAST ? 0 : kind == SLOW ? 1 : 2; out->llc_get[k] += llc.now() - l0; out->llc_n[k]++; }
            if (!g_lean) ck.verify(key, req, buf.data(), vlen, h, kind);
            if (sfl) { uint32_t gv; memcpy(&gv, buf.data() + 8, 4); ck.verify_stale(key, sfl, gv, h, kind); }
            if (ck.mono_on()) { uint32_t gv; memcpy(&gv, buf.data() + 8, 4); ck.verify_mono(h, key, mfl, gv, kind); }
            out->cpu_client_work_ns += (uint64_t)((c1 - c0) / g_tsc_ghz);
            if (g_ttl_side && kind == MISS && vlen) ttl_install(T, i, key, false);   // the miss refills the copy
            uint8_t held = 0;
            if (!g_held.empty()) {   // tracked in the warm-up too, so that "first read" means first in the run
                uint8_t &hd = g_held[h][key];
                held = hd;
                if (kind != MISS) hd = 1;
            }
            if (!measuring) { tl_miss_class = -1; continue; }
            if (g_open_rate > 0) {
                uint64_t done = now_ns();
                out->open_get.add(done > (uint64_t)t_arr ? done - (uint64_t)t_arr : 0);
                out->open_q.add(t_issue > (uint64_t)t_arr ? t_issue - (uint64_t)t_arr : 0);
            }
            uint64_t ns = (uint64_t)((c1 - c0) / g_tsc_ghz);
            out->gets++;
            out->all.add(ns);
            if (expired) { out->ttl_expired++; if (kind == MISS) out->ttl_miss++; }
            if (kind == MISS) {
                out->miss++;
                if (g_miss_diag) {
                    int c = tl_miss_class;
                    if (g_first_get[dkey(h, key)] == (uint32_t)i) c = MC_COLD;
                    if (c >= 0) out->mc[c]++;
                }
            } else { out->hit.add(ns); (kind == FAST ? out->fast : out->slow)++; (kind == FAST ? out->fasth : out->slowh).add(ns); }
            out->rk[cl.last_reason()][kind == FAST ? 0 : kind == SLOW ? 1 : 2]++;
            if (!g_held.empty() && cl.last_reason() == Client::R_NOENTRY) {
                if (held) (kind == MISS ? out->again_miss : out->again_slow)++;
                else (kind == MISS ? out->first_miss : out->first_slow)++;
            }
            if (kind == FAST && !g_lean) {   // ghost hit? (diagnostic, after the timed part)
                int ho;
                uint64_t off, aoff;
                uint32_t gen, agen;
                cl.last_hit(&ho, &off, &gen);
                if (!cache.ai_peek(key, &aoff, &agen)) out->ghost_absent++;
                else if (aoff != off || agen != gen) out->ghost_moved++;
            }
            if (!g_lean) {   // visibility tracking (diagnostic)
            uint32_t ver = 0;
            if (vlen >= 12) memcpy(&ver, buf.data() + 8, 4);
            if (ver > ck.begin(key)) out->early_vis++;
            uint32_t pv = V.ver[key].load(std::memory_order_acquire);
            if (pv && ver >= pv) {
                uint32_t bit = 1u << h;
                if (!(V.seen[key].load(std::memory_order_relaxed) & bit) && !(V.seen[key].fetch_or(bit) & bit)) {
                    uint64_t r = V.ret[key].load(std::memory_order_relaxed);
                    out->vis.add(c1 > r ? (uint64_t)((c1 - r) / g_tsc_ghz) : 0);
                }
            }
            }
        } else if (t == 4 && g_deletes) {   // a DELETE (as a PUT: through the pool, or in the client)
            if (g_ttl_side) g_ttl_exp[key].store(0, std::memory_order_relaxed);
            if (pool) {
                uint64_t s0 = __rdtsc();
                out->put_blocked_ns += pool->submit(key, measuring, W_DEL);
                out->cpu_client_work_ns += (uint64_t)((__rdtsc() - s0) / g_tsc_ghz);
                continue;
            }
            uint64_t c0 = __rdtsc();
            uint32_t v = cl.del(key);
            uint64_t c1 = __rdtsc();
            out->cpu_client_work_ns += (uint64_t)((c1 - c0) / g_tsc_ghz);
            ck.put_done(key, v);
            if (v) {
                V.ret[key].store(c1, std::memory_order_relaxed);
                V.seen[key].store(1u << h, std::memory_order_relaxed);
                V.ver[key].store(v, std::memory_order_release);
            }
            if (measuring) { if (v) out->dels++; else out->put_failed++; }
        } else if ((t == 1 || t == 3) && pool) {
            if (g_ttl_side) ttl_install(T, i, key, t == 3);
            uint64_t s0 = __rdtsc();
            out->put_blocked_ns += pool->submit(key, measuring);
            out->cpu_client_work_ns += (uint64_t)((__rdtsc() - s0) / g_tsc_ghz);
        } else if (t == 1 || t == 3) {
            if (g_ttl_side) ttl_install(T, i, key, t == 3);
            uint64_t c0 = __rdtsc();
            uint32_t v = cl.put(key);
            uint64_t c1 = __rdtsc();
            out->cpu_client_work_ns += (uint64_t)((c1 - c0) / g_tsc_ghz);
            ck.put_done(key, v);
            if (v) {   // publish for the visibility measurement: return time and the writer's bit first, version last
                V.ret[key].store(c1, std::memory_order_relaxed);
                V.seen[key].store(1u << h, std::memory_order_relaxed);
                V.ver[key].store(v, std::memory_order_release);
            }
            if (measuring) {
                if (v) { out->puts++; out->put.add((uint64_t)((c1 - c0) / g_tsc_ghz)); }
                else out->put_failed++;
            }
        }
    }
    out->wall_s = measuring ? (now_ns() - tw0) / 1e9 : 0;   // the GET client's time (PUTs still in flight not included)
    out->cpu_client_ns += thread_cpu_ns();
    if (pool) pool->finish(out);
    ClientStats e = cl.totals();
    out->r_noentry = e.slow_noentry - base.slow_noentry;
    out->r_logdel = e.slow_logdel - base.slow_logdel;
    out->r_hbstale = e.hb_stale - base.hb_stale;
    out->r_genmiss = e.fast_fail - base.fast_fail;
    out->r_dropall = e.dropgen_miss - base.dropgen_miss;
    out->r_lease = e.lease_expired - base.lease_expired;
    out->heat_reported = e.heat_reported - base.heat_reported;
}
// all client threads of host h; their sums are merged into *out
static void run_host_threads(Cache &cache, Checker &ck, int h, int hosts, const Trace &T, uint64_t nrec, uint64_t warm,
                             uint64_t seed, int nthr, bool bind, HostSum *out, const Vis &V) {
    std::vector<HostSum> part(nthr);
    std::vector<std::thread> th;
    for (int t = 0; t < nthr; t++)
        th.emplace_back([&, t] {
            if (bind && numa_available() >= 0) numa_run_on_node(h % (numa_max_node() + 1));
            run_host(cache, ck, h, hosts, T, nrec, warm, seed, t, nthr, &part[t], V);
        });
    for (auto &x : th) x.join();
    for (auto &p : part) out->merge(p);
}
// reason x outcome matrix as "none:fast/slow/miss,noentry:...,..."
static std::string rk_str(const HostSum &x) {
    static const char *rn[Client::R_N] = {"none", "noentry", "logdel", "hbstale", "genmiss", "dropall", "lease", "marked", "xdir"};
    std::string s;
    for (int a = 0; a < Client::R_N; a++) {
        char b[96];
        snprintf(b, sizeof b, "%s%s:%lu/%lu/%lu", a ? "," : "", rn[a], (unsigned long)x.rk[a][0], (unsigned long)x.rk[a][1],
                 (unsigned long)x.rk[a][2]);
        s += b;
    }
    return s;
}
static double g_time_scale = 1, g_trace_span_s = 0;
// Krater's committed ack (--put-order swap --put-wait 0, log scheme, not XTRA's own freshness) promises no
// read-after-write: the checker counts bounded staleness from the cut instead (Checker::on_cut / err_stale) and
// reports the GETs that returned an older version than an already acknowledged PUT as raw_misses, not as errors
static bool g_committed = false;
static uint64_t ck_errors(const Checker &c) { return c.errors() - (g_committed ? c.p2_errors() : 0); }
static void print_errs(const std::vector<std::string> &v, int host = -1) {
    for (auto &e : v) {
        const bool raw = g_committed && e.find(": P2:") != std::string::npos;
        if (host >= 0) printf("# %s (host %d): %s\n", raw ? "raw miss (committed ack: not counted)" : "error", host, e.c_str());
        else printf("# %s: %s\n", raw ? "raw miss (committed ack: not counted)" : "error", e.c_str());
    }
}
// shared-memory accesses per operation (whole run): shm_<op>[ops=N flush=f rd=r wr=w msg=m], f..m per operation
static std::string shm_line(Stats &S) {
    static const char *names[SHM_N] = {"fast", "slow", "miss", "fill", "put", "poll"};
    std::string s;
    char b[200];
    for (int k = 0; k < SHM_N; k++) {
        uint64_t n = S.shm[k][4].load();
        double d = n ? (double)n : 1.0;
        snprintf(b, sizeof b, "%sshm_%s[ops=%lu flush=%.3f rd=%.3f wr=%.3f msg=%.3f]", k ? " " : "", names[k], (unsigned long)n,
                 S.shm[k][0].load() / d, S.shm[k][1].load() / d, S.shm[k][2].load() / d, S.shm[k][3].load() / d);
        s += b;
    }
    return s;
}
// "lo:count" buckets of a latency histogram: 25 ns wide below 2000 ns, then [2^e, 2^(e+1))
static std::string hist_str(const Hist &h) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    for (int i = 0; i < Hist::NB; i++) {
        if (!h.b[i]) continue;
        double v = Hist::val(i);
        uint64_t lo = v < 2000 ? (uint64_t)(v / 25) * 25 : (1ULL << (63 - __builtin_clzll((uint64_t)v)));
        if (!out.empty() && out.back().first == lo) out.back().second += h.b[i];
        else out.push_back({lo, h.b[i]});
    }
    std::string s;
    char buf[48];
    for (auto &p : out) { snprintf(buf, sizeof buf, " %lu:%lu", (unsigned long)p.first, (unsigned long)p.second); s += buf; }
    return s;
}
// P6 (--mono 0|1): monotonic-read errors (whole run) and Config::mono's counters, summed over the hosts
static void print_mono(int hosts, const std::function<Stats &(int)> &st, const Checker &ck) {
    if (g_mono_arg < 0) return;
    uint64_t t = 0, tm = 0, b = 0, bm = 0, bk = 0, bs = 0, r = 0, o = 0, ob = 0, fd = 0, f = 0, sl = 0;
    for (int h = 0; h < hosts; h++) {
        Stats &x = st(h);
        t += x.mono_table; tm += x.mono_table_miss; b += x.mono_backend; bm += x.mono_bk_moved; bk += x.mono_bk_kg; bs += x.mono_bk_slow;
        r += x.mono_records; o += x.mono_overflow; ob += x.mono_ovf_backend; fd += x.mono_fd_overrun; f += x.get_fast; sl += x.get_slow;
    }
    // would-be hits = fast + slow + the backend reads Config::mono took instead of a hit (it only does so at a hit / slow check)
    const double wb = (double)(f + sl + b);
    printf("MONO mono=%d mono_errors=%lu gets_checked=%lu | whole run: table_gets=%lu (plain misses %lu) table_share_of_would_be_hits=%.6f "
           "backend_instead=%lu (%.6f of would-be hits; moved=%lu kg=%lu slow=%lu overflow=%lu) records=%lu overflow=%lu fd_overrun=%lu\n",
           g_mono_arg, (unsigned long)ck.mono_errors(), (unsigned long)ck.counters().mono_gets.load(), (unsigned long)t, (unsigned long)tm,
           wb ? (double)(t - tm) / wb : 0.0, (unsigned long)b, wb ? (double)b / wb : 0.0, (unsigned long)bm, (unsigned long)bk,
           (unsigned long)bs, (unsigned long)ob, (unsigned long)r, (unsigned long)o, (unsigned long)fd);
}
static void print_perf(int hosts, std::vector<HostSum> &hs, const char *tag, const Checker &ck, double pct, Scheme sc,
                       const std::function<Stats &(int)> &st, size_t log_lines) {
    HostSum tot;
    for (int h = 0; h < hosts; h++) {
        HostSum &x = hs[h];
        Stats &S = st(h);
        double pm = S.poll_cycles.load() / g_tsc_ghz / 1e6;
        printf("# host %d detail: not_fast[noentry=%lu logdel=%lu hbstale=%lu genmiss=%lu dropall=%lu lease=%lu] "
               "put_ns[mean=%.0f p50=%.0f p99=%.0f] put_failed=%lu visible_after_put_return_ns[n=%lu p50=%.0f p99=%.0f] "
               "early_visible=%lu | whole run: log_records_appended=%lu log_laps=%.3f polls=%lu poll_ms=%.1f "
               "poll_us_per_round=%.2f records_applied=%lu entries_deleted=%lu install_refused=%lu dropall=%lu overrun=%lu "
               "sweep_dropped=%lu restamps=%lu refreshed=%lu refresh_us_per_refresh=%.3f li_evicted=%lu li_count=%lu li_cap=%lu "
               "shard_remote=%lu shard_channel_slower_than_target=%lu (on misses %lu) slab_bump=%lu slab_free=%lu owner_used=%lu owner_limit=%lu xdir_reads=%lu xdir_other=%lu xdir_stale=%lu poll_cpu_ms=%.1f %s\n",
               h, (unsigned long)x.r_noentry, (unsigned long)x.r_logdel, (unsigned long)x.r_hbstale, (unsigned long)x.r_genmiss,
               (unsigned long)x.r_dropall, (unsigned long)x.r_lease, x.put.mean(), x.put.q(0.5), x.put.q(0.99),
               (unsigned long)x.put_failed, (unsigned long)x.vis.n, x.vis.q(0.5), x.vis.q(0.99), (unsigned long)x.early_vis,
               (unsigned long)S.log_records.load(), (double)S.log_records.load() / ((double)log_lines * 7),
               (unsigned long)S.polls.load(), pm, S.polls.load() ? 1000.0 * pm / S.polls.load() : 0,
               (unsigned long)S.log_applied.load(), (unsigned long)S.log_deleted.load(), (unsigned long)S.log_refused.load(),
               (unsigned long)S.log_dropall.load(), (unsigned long)S.log_overrun.load(), (unsigned long)S.sweep_dropped.load(),
               (unsigned long)S.restamps.load(), (unsigned long)S.log_refreshed.load(),
               S.log_refreshed.load() ? S.refresh_cycles.load() / g_tsc_ghz / 1e3 / S.log_refreshed.load() : 0.0,
               (unsigned long)S.li_evicted.load(), (unsigned long)S.li_count.load(), (unsigned long)S.li_cap.load(),
               (unsigned long)S.shard_remote.load(), (unsigned long)S.shard_overrun.load(),
               (unsigned long)S.shard_overrun_miss.load(), (unsigned long)S.slab_bump.load(),
               (unsigned long)S.slab_free.load(), (unsigned long)S.owner_used.load(), (unsigned long)S.owner_limit.load(),
               (unsigned long)S.xdir_reads.load(), (unsigned long)S.xdir_other.load(), (unsigned long)S.xdir_stale.load(),
               S.poll_cpu_ns.load() / 1e6,
               shm_line(S).c_str());
        printf("# host %d: gets=%lu hit=%.4f fast=%lu slow=%lu miss=%lu | GET ns mean=%.0f p50=%.0f p99=%.0f | hit ns "
               "mean=%.0f p50=%.0f p99=%.0f | %.2f MGET/s per client thread | puts=%lu put_ns_p50=%.0f | handler busy %.1f%%\n",
               h, (unsigned long)x.gets, x.gets ? 1.0 * (x.fast + x.slow) / x.gets : 0, (unsigned long)x.fast,
               (unsigned long)x.slow, (unsigned long)x.miss, x.all.mean(), x.all.q(0.5), x.all.q(0.99), x.hit.mean(),
               x.hit.q(0.5), x.hit.q(0.99), x.wall_s > 0 ? x.gets / x.wall_s / 1e6 : 0, (unsigned long)x.puts,
               x.put.q(0.5), x.wall_s > 0 ? 100 * x.busy_s / x.wall_s : 0);
        // per-host hit paths, so that the effective cost per hit follows from the prototype
        printf("# host %d paths: fast_ns[mean=%.1f p50=%.0f p99=%.0f] slow_ns[mean=%.1f p50=%.0f p99=%.0f] first_read[slow=%lu "
               "miss=%lu] noentry_again[slow=%lu miss=%lu] rk=%s\n",
               h, x.fasth.mean(), x.fasth.q(0.5), x.fasth.q(0.99), x.slowh.mean(), x.slowh.q(0.5), x.slowh.q(0.99),
               (unsigned long)x.first_slow, (unsigned long)x.first_miss, (unsigned long)x.again_slow,
               (unsigned long)x.again_miss, rk_str(x).c_str());
        // hit latency distribution (measured part), fast and slow path separately; buckets of 25 ns below 2 us, then
        // powers of two ("lo:count", empty buckets left out)
        printf("# host %d hit_hist fast:%s | slow:%s\n", h, hist_str(x.fasth).c_str(), hist_str(x.slowh).c_str());
        // CPU time by role, whole run (ms); process = 0 outside --procs
        printf("# host %d cpu: client_ms=%.1f putpool_ms=%.1f process_ms=%.1f poll_cpu_ms=%.1f client_work_ms=%.1f handler_busy_ms=%.1f\n",
               h, x.cpu_client_ns / 1e6, x.cpu_put_ns / 1e6, x.cpu_proc_ns / 1e6, S.poll_cpu_ns.load() / 1e6,
               x.cpu_client_work_ns / 1e6, S.handler_busy_cycles.load() / g_tsc_ghz / 1e6);
        if (g_llc && h == g_llc_host)
            printf("# host %d llc: mode=%s fast=%lu/%lu slow=%lu/%lu miss=%lu/%lu proc_whole=%lu gets_whole=%lu\n", h,
                   x.llc_mode == 1 ? "rdpmc" : x.llc_mode == 2 ? "read" : "off", (unsigned long)x.llc_get[0], (unsigned long)x.llc_n[0],
                   (unsigned long)x.llc_get[1], (unsigned long)x.llc_n[1], (unsigned long)x.llc_get[2], (unsigned long)x.llc_n[2],
                   (unsigned long)x.llc_proc, (unsigned long)x.gets_whole);
        if (g_open_rate > 0)
            printf("# host %d open_loop: rate=%.0f offered=%lu span_s=%.3f achieved_rate=%.0f get_ns[mean=%.0f p50=%.0f p99=%.0f] "
                   "queue_ns[mean=%.0f p50=%.0f p99=%.0f]\n", h, g_open_rate, (unsigned long)x.open_recs, x.open_span_ns / 1e9,
                   x.wall_s > 0 ? x.open_recs / x.wall_s : 0.0, x.open_get.mean(), x.open_get.q(0.5), x.open_get.q(0.99),
                   x.open_q.mean(), x.open_q.q(0.5), x.open_q.q(0.99));
        if (g_miss_diag) {
            printf("# host %d miss_diag: misses=%lu", h, (unsigned long)x.miss);
            for (int c = 0; c < MC_N; c++) printf(" %s=%lu", MC_NAMES[c], (unsigned long)x.mc[c]);
            printf("\n");
        }
        printf("# host %d li_clock: queue=%lu dedup_passes=%lu (Config::li_clock_inq)\n", h, (unsigned long)S.li_clock_q.load(),
               (unsigned long)S.li_clock_dedups.load());
        printf("# host %d fills: admits=%lu drop_tok=%lu drop_pending=%lu drop_cached=%lu drop_space=%lu drop_restart=%lu "
               "ring_drop=%lu evictions=%lu tok_timeout=%lu\n", h, (unsigned long)S.admits.load(), (unsigned long)S.fill_drop_tok.load(),
               (unsigned long)S.fill_drop_pending.load(), (unsigned long)S.fill_drop_cached.load(), (unsigned long)S.fill_drop_space.load(),
               (unsigned long)S.fill_drop_restart.load(), (unsigned long)S.drop_ring_full.load(), (unsigned long)S.evictions.load(),
               (unsigned long)S.tok_timeout.load());
        if (S.diag_put_sum[9].load() + S.diag_put_sum[4].load()) {   // --diag-put
            static const char *seg[10] = {"send_to_owner", "owner_queue", "unpublish", "append_cut", "backend_write",
                                          "publish", "wait_W", "late_after_due", "reply_transit", "whole_put"};
            printf("# host %d diag_put (whole run; log2 buckets, quantiles are upper bucket edges, us):", h);
            for (int k = 0; k < 10; k++) {
                uint64_t n = 0, c = 0;
                for (int b = 0; b < 48; b++) n += S.diag_put_hist[k][b].load();
                double p50 = 0, p99 = 0;
                for (int b = 0; b < 48; b++) {
                    c += S.diag_put_hist[k][b].load();
                    if (n && !p50 && c >= 0.5 * n) p50 = (double)(1ULL << b) / g_tsc_ghz / 1e3;
                    if (n && !p99 && c >= 0.99 * n) p99 = (double)(1ULL << b) / g_tsc_ghz / 1e3;
                }
                printf(" %s[n=%lu mean=%.1f p50<=%.1f p99<=%.1f]", seg[k], (unsigned long)n,
                       n ? S.diag_put_sum[k].load() / g_tsc_ghz / 1e3 / n : 0.0, p50, p99);
            }
            printf("\n");
        }
        if (S.put_async_puts.load() + S.reply_early.load())   // Config::put_async: reply_early must be 0
            printf("# host %d put_async: puts=%lu reply_early=%lu\n", h, (unsigned long)S.put_async_puts.load(),
                   (unsigned long)S.reply_early.load());
        if (S.diag_hist[1][0].load() + S.diag_sum[1].load()) {   // --diag-lock
            auto q = [&](int k, double p) {   // quantile from the log2 histogram (upper bucket edge), ns
                uint64_t n = 0, c = 0;
                for (int b = 0; b < 48; b++) n += S.diag_hist[k][b].load();
                for (int b = 0; b < 48; b++) { c += S.diag_hist[k][b].load(); if (n && c >= p * n) return (double)(1ULL << b) / g_tsc_ghz; }
                return 0.0;
            };
            auto cnt = [&](int k) { uint64_t n = 0; for (int b = 0; b < 48; b++) n += S.diag_hist[k][b].load(); return n; };
            auto mean = [&](int k) { uint64_t n = cnt(k); return n ? S.diag_sum[k].load() / g_tsc_ghz / n : 0.0; };
            printf("# host %d diag_lock (whole run, log2 buckets: quantiles are upper bucket edges): slow_path[n=%lu mean_ns=%.0f "
                   "p50<=%.0f p90<=%.0f p99<=%.0f] poll_mu_wait[n=%lu mean_ns=%.0f p50<=%.0f p90<=%.0f p99<=%.0f share_of_slow=%.3f] "
                   "poll_mu_hold_per_round[n=%lu mean_ns=%.0f p50<=%.0f p90<=%.0f p99<=%.0f] "
                   "stripe_wait[n=%lu mean_ns=%.0f p50<=%.0f p99<=%.0f share_of_slow=%.3f]\n", h,
                   (unsigned long)cnt(1), mean(1), q(1, .5), q(1, .9), q(1, .99), (unsigned long)cnt(0), mean(0), q(0, .5), q(0, .9),
                   q(0, .99), S.diag_sum[1].load() ? (double)S.diag_sum[0].load() / S.diag_sum[1].load() : 0.0,
                   (unsigned long)cnt(2), mean(2), q(2, .5), q(2, .9), q(2, .99), (unsigned long)cnt(3), mean(3), q(3, .5),
                   q(3, .99), S.diag_sum[1].load() ? (double)S.diag_sum[3].load() / S.diag_sum[1].load() : 0.0);
        }
        if (g_put_inflight > 0)
            printf("# host %d async_put: inflight=%d puts=%lu put_ns[mean=%.0f p50=%.0f p99=%.0f] queue_wait_ns[p50=%.0f p99=%.0f] "
                   "queue_max=%lu client_blocked_ms=%.1f (%.2f%% of the GET client's time)\n",
                   h, g_put_inflight, (unsigned long)x.puts, x.put.mean(), x.put.q(0.5), x.put.q(0.99), x.putq.q(0.5),
                   x.putq.q(0.99), (unsigned long)x.put_queue_max, x.put_blocked_ns / 1e6,
                   x.wall_s > 0 ? 100.0 * x.put_blocked_ns / 1e9 / x.wall_s : 0);
        if (g_deletes || g_ttl_side)   // only with --deletes 1 / --ttl-side
            printf("# host %d writes: deletes=%lu ttl_expired=%lu ttl_expired_then_miss=%lu ttl_expired_whole=%lu | owner (whole "
                   "run): deletes=%lu expires=%lu expire_dropped=%lu | get_absent=%lu\n", h, (unsigned long)x.dels,
                   (unsigned long)x.ttl_expired, (unsigned long)x.ttl_miss, (unsigned long)x.ttl_expired_whole,
                   (unsigned long)S.deletes.load(), (unsigned long)S.expires.load(), (unsigned long)S.expire_dropped.load(),
                   (unsigned long)S.get_absent.load());
        tot.merge(x);
    }
    printf("RESULT %s scheme=%s pct=%g gets=%lu hit_ratio=%.6f get_ns_mean=%.1f get_ns_p50=%.0f get_ns_p99=%.0f "
           "hit_ns_mean=%.1f hit_ns_p50=%.0f hit_ns_p99=%.0f errors=%lu puts=%lu put_ns_p50=%.0f put_ns_p99=%.0f "
           "put_failed=%lu not_fast_noentry=%lu not_fast_logdel=%lu not_fast_hbstale=%lu not_fast_genmiss=%lu "
           "not_fast_dropall=%lu not_fast_lease=%lu visible_ns_p50=%.0f visible_ns_p99=%.0f early_visible=%lu "
           "ghost_absent=%lu ghost_moved=%lu noentry_again_slow=%lu noentry_again_miss=%lu heat_reports_per_get=%.4f rk=%s%s\n",
           tag, scheme_name(sc), pct, (unsigned long)tot.gets, tot.gets ? 1.0 * (tot.fast + tot.slow) / tot.gets : 0,
           tot.all.mean(), tot.all.q(0.5), tot.all.q(0.99), tot.hit.mean(), tot.hit.q(0.5), tot.hit.q(0.99),
           (unsigned long)ck_errors(ck), (unsigned long)tot.puts, tot.put.q(0.5), tot.put.q(0.99), (unsigned long)tot.put_failed,
           (unsigned long)tot.r_noentry, (unsigned long)tot.r_logdel, (unsigned long)tot.r_hbstale, (unsigned long)tot.r_genmiss,
           (unsigned long)tot.r_dropall, (unsigned long)tot.r_lease, tot.vis.q(0.5), tot.vis.q(0.99), (unsigned long)tot.early_vis,
           (unsigned long)tot.ghost_absent, (unsigned long)tot.ghost_moved, (unsigned long)tot.again_slow,
           (unsigned long)tot.again_miss, tot.gets ? (double)tot.heat_reported / tot.gets : 0.0, rk_str(tot).c_str(),
           g_committed ? (" committed=1 raw_misses=" + std::to_string(ck.p2_errors()) + " stale_errors=" +
                          std::to_string(ck.stale_errors())).c_str() : "");
    if (g_deletes || g_ttl_side) {   // measured part (these counts are not in the RESULT line above)
        uint64_t ga = 0, xd = 0;
        for (int h = 0; h < hosts; h++) { ga += st(h).get_absent.load(); xd += st(h).expire_dropped.load(); }
        // expire_dropped_whole: EXPIREs that removed a cached copy (PRIV: copies dropped on all hosts), whole run; the rest of
        // ttl_expired_whole found the key no longer cached (evicted: its GET is a capacity miss anyway)
        printf("WRITES deletes=%lu ttl_side=%s ttl_expired=%lu ttl_expired_per_get=%.6f ttl_expired_then_miss=%lu "
               "ttl_expired_whole=%lu expire_dropped_whole=%lu get_absent_whole=%lu\n", (unsigned long)tot.dels,
               g_ttl_side ? g_ttl_side_f.c_str() : "-", (unsigned long)tot.ttl_expired,
               tot.gets ? (double)tot.ttl_expired / tot.gets : 0.0, (unsigned long)tot.ttl_miss, (unsigned long)tot.ttl_expired_whole,
               (unsigned long)xd, (unsigned long)ga);
    }
    print_mono(hosts, st, ck);
    // trace time vs wall time of the measured part
    printf("TIME time_scale=%g pace=%s trace_span_s=%.0f wall_s=%.2f compression=%.1f\n", g_time_scale,
           g_pace.k > 0 ? (g_spread.empty() ? "trace" : "spread") : "free", g_trace_span_s, tot.wall_s, tot.wall_s > 0 ? g_trace_span_s / tot.wall_s : 0.0);
}

static void usage() {
    fprintf(stderr,
            "usage: replay --trace <run.binary> --objsize <objsize.bin> --pct <M%%> [--hosts 4] [--seed 1]\n"
            "  [--heat 0|B] [--accounting sim|full|pool (only the shared pool is charged; PRIV gets the local index DRAM extra)] [--limit N] [--delay-ns 0] [--expect <sim hit ratio>]\n"
            "  [--report-every 10000000] [--slab-factor 4] [--slab-pad-mb 1] [--perf [--procs] [--bind] [--shm-node N] [--warm-frac 0.1]] [--scheme rackcache|cxl_uc|ver|ver_noflush|gstate_cxl|gstate_coh|priv|shard|xtra] [--rpc-ns 0]\n"
            "  [--update log|lease|invalidate (default log)] [--poll-us 1000 (log P)] [--lease-us 1000 (lease T)]\n"
            "  [--owner-salt <u64> (key -> owner hash salt, default 0)]\n"
            "  [--bin <records.bin> (instead of --trace: u64 records with client ids, tools/ycsb/trace_io.h)] [--routing random|client]\n"
            "  [--skip N] [--threads T (per host, --perf)] [--rpc-dist <dir>:<kind>:<client> (SHARD)] [--xtra-dir-slots 1048576 (xtra)] [--xtra-own-freshness 0|1 (xtra checks its directory word, no log / poller / heartbeat)] [--xtra-dir-replicas 0|1|2 (with own freshness, a directory copy per host; 1: clwb, 2: clflushopt)] [--xtra-hk-us 10000] [--swap-refresh-after-del 1|0 (swap: 1 = delete record then refresh, 0 = refresh then delete)] [--li32 (32 B local-index entries)] [--li-slots N (auto)] [--hit-clock lfence3|rdtscp] [--obj-read copy|prefetch|mstore|both] [--miss-diag (threads mode)] [--diag-lock] [--diag-put] [--exec-slack-ns 1] [--exec-lanes 4] [--charge k11b|layout] [--m-base run|objsize] [--put-async 0|1] [--put-idle-ns 0] [--handler-idle-us 20] [--put-key-after reply|publish] [--put-pool-away] [--open-rate R (open loop, Poisson arrivals at R records/s per host, GET latency incl. queueing; not with --pace)] [--no-puts (skip write records)] [--llc-count (LLC misses per GET on host 1)] [--gen-bits 32] [--sweep-slots N (auto)] [--deletes 0|1 (op 4 runs as a DELETE)] [--ttl-side <u32 per record> (TTL expiry, --bin, not --procs)] [--ro-peers 0|1 (other hosts' blocks mapped read-only; --perf --procs)] [--mono 0|1 (monotonic reads; P6 checked)] [--mono-hold 0|1] [--mono-fd-us 10000]\n"
            "  [--feed 1|0 (1 change feed / 0 lazy log)] [--kg cxl|msg] [--li-cap 1 (x objects; 0 = unlimited)]\n"
            "  [--heat-dedup 1|0] [--heat-window-polls 1] [--shard-direct (threads mode)] [--keep-failed 0|1]\n"
            "  [--put-inflight K (--perf: PUTs to K writer threads per client, GET metrics exclude PUT waits; 0 = closed loop)]\n"
            "  [--backend-get-dist <table> --backend-put-dist <table> (sampled backend latency)]\n"
            "  [--time-scale k (P, W, T divided by k)] [--pace free|trace|spread (issue at trace time / k; --bin)] [--write-delay-ns]\n");
    exit(2);
}

int main(int argc, char **argv) {
    std::string trace, objsize;
    double pct = -1, expect = -1;
    int hosts = 4, heat = 0;
    uint64_t seed = 1, limit = ~0ULL, delay_ns = 0, report_every = 10000000;
    double slab_factor = 4, slab_pad_mb = 1;
    bool sim_acct = true, pool_acct = false;
    Scheme scheme = Scheme::RACKCACHE;
    uint64_t rpc_ns = 0;
    bool perf = false, procs = false, bind = false, ro_peers = false;
    uint64_t mono_fd_ns = 0;   // --mono-fd-us: Config::mono_fd_ns (0 = Config's)
    int shm_node = -1;
    double warm_frac = 0.1;
    Rules rules;   // default update mechanism: log
    uint64_t poll_us = 0, lease_us = 0, skip = 0;
    std::string binf, routing = "random", rpc_dist_spec;
    int nthr = 1;
    double li_cap = -1;
    int heat_dedup = -1, heat_window_polls = 0;
    bool shard_direct = false;
    size_t xtra_dir_slots = 1 << 20;   // --xtra-dir-slots: Config::xtra_dir_slots
    bool xtra_own = false;             // --xtra-own-freshness: Config::xtra_own_freshness
    int xtra_rep = 0;             // --xtra-dir-replicas (XREP): Config::xtra_dir_replicas
    uint64_t xtra_hk_ns = 10'000'000;
    bool li16 = true;                  // 16 B local-index entries (default; --li32 = 32 B)
    int hit_clock = 1;                 // --hit-clock lfence3|rdtscp (rdtscp is the default)
    int obj_read = -1;                 // --obj-read copy|prefetch|mstore|both (default: Config's)
    bool diag_lock = false;            // --diag-lock: Config::diag_lock_timing (slow path / poll_mu histograms)
    bool li_clock_inq = true;          // --li-clock-inq 0|1: Config::li_clock_inq
    bool diag_put = false;             // --diag-put: Config::diag_put_timing (PUT segment histograms)
    int64_t exec_slack_ns = -1;        // --exec-slack-ns: Config::exec_timer_slack_ns (0 = Linux default 50 us; -1 = Config's)
    int charge_model = -1;             // --charge k11b|layout: Config::charge_model (-1 = Config's: k11b)
    int exec_lanes = 0;                // --exec-lanes: Config::exec_lanes (0 = Config's: 4)
    int put_async = -1;                // --put-async 0|1: Config::put_async (-1 = Config's)
    int64_t put_idle_ns = -1;          // --put-idle-ns: Config::put_idle_ns (-1 = Config's)
    int handler_idle_us = -1;          // --handler-idle-us: Config::handler_idle_us (-1 = Config's)
    int put_key_after = -1;            // --put-key-after reply|publish: Config::put_key_after_publish (-1 = Config's)
    bool m_base_objsize = false;       // --m-base run|objsize: M = pct of the run's keys' bytes (default) / of all of objsize
    uint64_t li_slots_set = 0;         // --li-slots: local-index slots per host (0 = sized from the cap, see below)
    int gen_bits = 0;                  // 0 = Config default (32)
    size_t sweep_slots = 0;            // --sweep-slots: generation-wrap sweep slots per poll (Config default otherwise)
    bool sweep_set = false;
    double time_scale = 1;
    bool pace_trace = false, pace_spread = false;
    uint64_t write_delay_ns = ~0ULL;
    std::string get_dist_f = "-", put_dist_f = "-";
    static option opts[] = {{"trace", 1, 0, 't'}, {"objsize", 1, 0, 'o'}, {"pct", 1, 0, 'p'}, {"hosts", 1, 0, 'H'},
                            {"seed", 1, 0, 's'}, {"heat", 1, 0, 'h'}, {"accounting", 1, 0, 'a'}, {"limit", 1, 0, 'l'},
                            {"delay-ns", 1, 0, 'd'}, {"expect", 1, 0, 'e'}, {"report-every", 1, 0, 'r'},
                            {"slab-factor", 1, 0, 'f'}, {"scheme", 1, 0, 'C'}, {"rpc-ns", 1, 0, 'Q'}, {"perf", 0, 0, 'P'}, {"procs", 0, 0, 'X'},
                            {"bind", 0, 0, 'B'}, {"shm-node", 1, 0, 'n'}, {"warm-frac", 1, 0, 'w'},
                            {"update", 1, 0, 'U'}, {"poll-us", 1, 0, 'Y'}, {"lease-us", 1, 0, 'L'},
                            {"bin", 1, 0, 'b'}, {"routing", 1, 0, 'R'}, {"skip", 1, 0, 'k'}, {"threads", 1, 0, 'T'},
                            {"rpc-dist", 1, 0, 'D'}, {"slab-pad-mb", 1, 0, 'Z'}, {"heat-dedup", 1, 0, 'J'},
                            {"heat-window-polls", 1, 0, 'W'}, {"shard-direct", 0, 0, 'V'}, {"keep-failed", 1, 0, 'K'}, {"put-inflight", 1, 0, 'j'},
                            {"time-scale", 1, 0, 'S'}, {"pace", 1, 0, 'A'}, {"write-delay-ns", 1, 0, 'E'}, {"feed", 1, 0, 'F'}, {"kg", 1, 0, 'G'}, {"li-cap", 1, 0, 'I'}, {"xtra-dir-slots", 1, 0, 300}, {"li16", 0, 0, 301}, {"gen-bits", 1, 0, 302}, {"sweep-slots", 1, 0, 303}, {"li32", 0, 0, 304}, {"li-slots", 1, 0, 305}, {"hit-clock", 1, 0, 306}, {"backend-get-dist", 1, 0, 307}, {"backend-put-dist", 1, 0, 308}, {"obj-read", 1, 0, 320}, {"miss-diag", 0, 0, 321}, {"diag-lock", 0, 0, 322}, {"put-pool-away", 0, 0, 323}, {"diag-put", 0, 0, 324}, {"exec-slack-ns", 1, 0, 325}, {"charge", 1, 0, 326}, {"exec-lanes", 1, 0, 327}, {"m-base", 1, 0, 328}, {"put-async", 1, 0, 329}, {"put-idle-ns", 1, 0, 330}, {"handler-idle-us", 1, 0, 331}, {"put-key-after", 1, 0, 332}, {"open-rate", 1, 0, 340}, {"no-puts", 0, 0, 341}, {"lean", 0, 0, 380}, {"llc-count", 0, 0, 342}, {"owner-salt", 1, 0, 343}, {"put-order", 1, 0, 344}, {"put-wait", 1, 0, 345}, {"xtra-own-freshness", 1, 0, 346}, {"xtra-dir-replicas", 1, 0, 372}, {"xtra-hk-us", 1, 0, 347}, {"swap-refresh-after-del", 1, 0, 348}, {"deletes", 1, 0, 349}, {"ttl-side", 1, 0, 350}, {"ro-peers", 1, 0, 351}, {"mono", 1, 0, 352}, {"mono-hold", 1, 0, 353}, {"mono-fd-us", 1, 0, 354}, {"li-clock-inq", 1, 0, 371}, {0, 0, 0, 0}};
    int ch;
    while ((ch = getopt_long(argc, argv, "", opts, nullptr)) != -1) {
        switch (ch) {
        case 't': trace = optarg; break;
        case 'o': objsize = optarg; break;
        case 'p': pct = atof(optarg); break;
        case 'H': hosts = atoi(optarg); break;
        case 's': seed = strtoull(optarg, 0, 0); break;
        case 'h': heat = atoi(optarg); break;
        case 'a': sim_acct = std::string(optarg) == "sim"; pool_acct = std::string(optarg) == "pool"; break;
        case 'l': limit = strtoull(optarg, 0, 0); break;
        case 'd': delay_ns = strtoull(optarg, 0, 0); break;
        case 'e': expect = atof(optarg); break;
        case 'r': report_every = strtoull(optarg, 0, 0); break;
        case 'f': slab_factor = atof(optarg); break;
        case 'C': if (!scheme_parse(optarg, &scheme)) usage(); break;
        case 'Q': rpc_ns = strtoull(optarg, 0, 0); break;
        case 'P': perf = true; break;
        case 'X': procs = true; break;
        case 'B': bind = true; break;
        case 'n': shm_node = atoi(optarg); break;
        case 'w': warm_frac = atof(optarg); break;
        case 'U': if (!set_update(rules, optarg)) usage(); break;
        case 'Y': poll_us = strtoull(optarg, 0, 0); break;
        case 'L': lease_us = strtoull(optarg, 0, 0); break;
        case 'b': binf = optarg; break;
        case 'R': routing = optarg; if (routing != "random" && routing != "client") usage(); break;
        case 'k': skip = strtoull(optarg, 0, 0); break;
        case 'T': nthr = std::max(1, atoi(optarg)); break;
        case 'D': rpc_dist_spec = optarg; break;
        case 'Z': slab_pad_mb = atof(optarg); break;
        case 'J': heat_dedup = atoi(optarg); break;
        case 'W': heat_window_polls = atoi(optarg); break;
        case 'V': shard_direct = true; break;
        case 300: xtra_dir_slots = strtoull(optarg, 0, 0); break;
        case 301: li16 = true; break;   // --li16: already the default (accepted, no effect)
        case 304: li16 = false; break;
        case 305: li_slots_set = strtoull(optarg, 0, 0); break;
        case 320: if ((obj_read = obj_read_mode_parse(optarg)) < 0) usage(); break;
        case 321: g_miss_diag = true; break;
        case 322: diag_lock = true; break;
        case 323: g_put_pool_away = true; break;
        case 324: diag_put = true; break;
        case 371: li_clock_inq = atoi(optarg) != 0; break;   // 0 = no in-queue mark (duplicate pushes, dedup)
        case 340: g_open_rate = atof(optarg); if (g_open_rate < 0) usage(); break;
        case 341: g_no_puts = true; break;
        case 380: g_lean = true; break;
        case 342: g_llc = true; break;
        case 343: setenv("RC_OWNER_SALT", optarg, 1); break;   // key -> owner hash salt (Cache::owner_of_static)
        // --put-order window (default: Krater's RAW by the window) | swap (Rules::put_cut_at_publish); --put-wait 1
        // (default: W after the cut) | 0 (reply at once: the baselines' own PUT, or Krater's committed ack).
        // Krater committed = swap + put-wait 0; RAW by waiting = swap + put-wait 1.
        case 344: if (std::string(optarg) == "swap") rules.put_cut_at_publish = true; else if (std::string(optarg) == "window") rules.put_cut_at_publish = false; else usage(); break;
        case 345: rules.put_wait = atoi(optarg) != 0; break;
        // --xtra-own-freshness 1 = XTRA-style checks freshness with its directory word (no Krater log, poller or heartbeat;
        // swap order, no W); 0 (default) = the borrowed variant (Krater's log, poller and heartbeat). --xtra-hk-us: its
        // housekeeping round
        case 346: xtra_own = atoi(optarg) != 0; break;
        case 372: {
            std::string v = optarg;
            if (v != "0" && v != "1" && v != "2") { fprintf(stderr, "--xtra-dir-replicas 0|1|2\n"); return 2; }
            xtra_rep = v[0] - '0';
            break;
        }
        case 347: xtra_hk_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 348: rules.swap_refresh_after_del = atoi(optarg) != 0; break;   // swap records: 1 delete, refresh (default); 0 refresh, delete
        case 349: g_deletes = atoi(optarg) != 0; break;   // op 4 runs as a DELETE
        case 350: g_ttl_side_f = optarg; break;           // TTL expiry from a per-record side file
        case 351: ro_peers = atoi(optarg) != 0; break;     // Config::ro_peers: other hosts' blocks read-only (--procs)
        case 352: g_mono_arg = atoi(optarg) != 0; break;     // Config::mono (monotonic reads); P6 checked when given
        case 353: rules.mono_hold = atoi(optarg) != 0; break;   // ablation: 0 = Config::mono's entries expire at once (M1-ghost)
        case 354: mono_fd_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 325: exec_slack_ns = strtoll(optarg, 0, 0); break;
        case 326:
            if (!strcmp(optarg, "k11b")) charge_model = CHARGE_K11B;
            else if (!strcmp(optarg, "layout")) charge_model = CHARGE_LAYOUT;
            else usage();
            break;
        case 327: exec_lanes = atoi(optarg); break;
        case 329: put_async = atoi(optarg); break;
        case 330: put_idle_ns = strtoll(optarg, 0, 0); break;
        case 331: handler_idle_us = atoi(optarg); break;
        case 332:
            if (!strcmp(optarg, "publish")) put_key_after = 1;
            else if (!strcmp(optarg, "reply")) put_key_after = 0;
            else usage();
            break;
        case 328:
            if (!strcmp(optarg, "objsize")) m_base_objsize = true;
            else if (!strcmp(optarg, "run")) m_base_objsize = false;
            else usage();
            break;
        case 306: if (std::string(optarg) == "rdtscp") hit_clock = 1; else if (std::string(optarg) == "lfence3") hit_clock = 0; else usage(); break;
        case 302: gen_bits = atoi(optarg); break;
        case 303: sweep_slots = strtoull(optarg, 0, 0); sweep_set = true; break;
        case 'K': rules.keep_failed = atoi(optarg) != 0; break;
        case 'j': g_put_inflight = std::max(0, atoi(optarg)); break;
        case 'S': time_scale = atof(optarg); if (time_scale <= 0) usage(); break;
        case 'A':
            if (std::string(optarg) == "trace") pace_trace = true;
            else if (std::string(optarg) == "spread") pace_trace = pace_spread = true;
            else if (std::string(optarg) != "free") usage();
            break;
        case 'E': write_delay_ns = strtoull(optarg, 0, 0); break;
        case 307: if (!(g_get_dist = load_lat_table(optarg))) return 2; get_dist_f = optarg; break;
        case 308: if (!(g_put_dist = load_lat_table(optarg))) return 2; put_dist_f = optarg; break;
        case 'F': rules.feed = atoi(optarg) != 0; break;
        case 'G': if (std::string(optarg) == "cxl") rules.kg_cxl = true; else if (std::string(optarg) == "msg") rules.kg_cxl = false; else usage(); break;
        case 'I': li_cap = atof(optarg); break;
        default: usage();
        }
    }
    if ((trace.empty() == binf.empty()) || objsize.empty() || pct <= 0) usage();
    if (routing == "client" && binf.empty()) { fprintf(stderr, "--routing client needs --bin (client ids)\n"); return 2; }
    // ---- inputs
    const std::string &tf = binf.empty() ? trace : binf;
    const size_t recsz = binf.empty() ? sizeof(Rec) : 8;
    int fd = open(tf.c_str(), O_RDONLY);
    struct stat sb;
    if (fd < 0 || fstat(fd, &sb)) { perror(tf.c_str()); return 1; }
    uint64_t total_recs = sb.st_size / recsz;
    if (skip > total_recs) skip = total_recs;
    uint64_t nrec = std::min<uint64_t>(total_recs - skip, limit);
    const char *base = (const char *)mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap trace"); return 1; }
    madvise((void *)(base + skip * recsz), nrec * recsz, MADV_SEQUENTIAL);
    Trace T;
    if (binf.empty()) T.R = (const Rec *)base + skip;
    else T.B = (const uint64_t *)base + skip;
    T.by_client = routing == "client";
    if (!g_ttl_side_f.empty()) {   // --ttl-side: u32 per record of the whole --bin file
        if (binf.empty() || procs) { fprintf(stderr, "--ttl-side needs --bin (trace seconds) and threads / serial mode (no --procs)\n"); return 2; }
        int tfd = open(g_ttl_side_f.c_str(), O_RDONLY);
        struct stat tsb;
        if (tfd < 0 || fstat(tfd, &tsb)) { perror(g_ttl_side_f.c_str()); return 1; }
        if ((uint64_t)tsb.st_size != total_recs * 4) {
            fprintf(stderr, "--ttl-side %s: %lu bytes, want 4 x %lu records of %s\n", g_ttl_side_f.c_str(), (unsigned long)tsb.st_size,
                    (unsigned long)total_recs, tf.c_str());
            return 2;
        }
        const char *tb = (const char *)mmap(nullptr, tsb.st_size, PROT_READ, MAP_SHARED, tfd, 0);
        if (tb == MAP_FAILED) { perror("mmap ttl side"); return 1; }
        g_ttl_side = (const uint32_t *)tb + skip;
    }
    FILE *f = fopen(objsize.c_str(), "rb");
    if (!f) { perror(objsize.c_str()); return 1; }
    std::vector<uint32_t> size;
    {
        uint32_t x;
        while (fread(&x, 4, 1, f) == 1) size.push_back(x);
        fclose(f);
    }
    uint64_t nkeys = size.size() - 1;
    if (g_ttl_side) g_ttl_exp.reset(new std::atomic<uint32_t>[nkeys + 1]());   // every key: no cached copy yet
    // pass 0: total distinct bytes of the keys in the (possibly truncated) run trace
    std::vector<uint8_t> seen(nkeys + 1, 0);
    uint64_t total = 0, maxkey = 0, ndistinct = 0;
    for (uint64_t i = 0; i < nrec; i++) {
        uint64_t k = T.key(i);
        if (k == 0 || k > nkeys) { fprintf(stderr, "key %lu out of objsize range\n", (unsigned long)k); return 1; }
        if (!seen[k]) { seen[k] = 1; total += size[k]; ndistinct++; }
        maxkey = std::max(maxkey, k);
    }
    {   // the pool charge per object under the chosen model (CHARGE_K11B: compare with the simulators' bytes per object)
        const bool k11b = (charge_model < 0 ? Config{}.charge_model : charge_model) == CHARGE_K11B;
        auto ch = [&](uint32_t v) -> double {
            v = std::min<uint32_t>(std::max<uint32_t>(v, 1), MAX_VLEN);
            if (k11b) return (double)charge_k11b(scheme, v);
            if (scheme == Scheme::PRIV || scheme == Scheme::SHARD) return v + HDR;
            static const uint32_t cls[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192};   // rackcache.cc CLS
            uint32_t nl = obj_lines(v), i = 0;
            while (cls[i] < nl) i++;
            return (double)cls[i] * LINE;
        };
        double wsum = 0, psum = 0; uint64_t wn = 0;
        for (uint64_t k = 1; k <= nkeys; k++) if (size[k]) { wsum += ch(size[k]); wn++; if (seen[k]) psum += ch(size[k]); }
        fprintf(stderr, "# replay: charge=%s pool bytes per object: %.1f (all %lu objects of objsize) %.1f (%lu keys of the run)\n",
                k11b ? "k11b" : "layout", wn ? wsum / wn : 0.0, (unsigned long)wn, ndistinct ? psum / ndistinct : 0.0,
                (unsigned long)ndistinct);
    }
    // --m-base objsize: M = pct of the distinct bytes of every key in objsize, as the simulators take it (on a prefix of a
    // trace window this is the window's bytes, not the prefix's); default run: pct of the distinct bytes of the keys in
    // the run
    uint64_t mbase = total;
    if (m_base_objsize) { mbase = 0; for (uint64_t k = 1; k <= nkeys; k++) mbase += size[k]; }
    uint64_t M = (uint64_t)((double)mbase * pct / 100.0);   // the simulators use integer pct: total*pct/100
    if (pct == (int)pct) M = mbase * (uint64_t)pct / 100;
    if (m_base_objsize) fprintf(stderr, "# replay: m_base=objsize (%lu bytes; the run's keys: %lu) M=%lu\n", (unsigned long)mbase,
                                (unsigned long)total, (unsigned long)M);
    uint64_t per_owner = M / hosts;
    // ---- cache
    if (g_get_dist) delay_ns = 0;
    if (g_put_dist) write_delay_ns = 0;
    else if (g_get_dist && write_delay_ns == ~0ULL) { fprintf(stderr, "--backend-get-dist needs --write-delay-ns or --backend-put-dist\n"); return 2; }
    if (g_get_dist || g_put_dist)
        fprintf(stderr, "# replay: backend get dist %s (mean %.0f ns), put dist %s (mean %.0f ns)\n", get_dist_f.c_str(),
                g_get_dist ? g_get_dist->mean : 0.0, put_dist_f.c_str(), g_put_dist ? g_put_dist->mean : 0.0);
    DistBackend be(nkeys, delay_ns);
    if (write_delay_ns != ~0ULL) be.set_put_delay(write_delay_ns);
    be.set_sizes(size.data());
    Checker ck(nkeys, &be);
    if (g_mono_arg >= 0) ck.enable_mono(hosts);
    Config cfg;
    cfg.hosts = hosts;
    cfg.scheme = scheme;
    cfg.rpc_ns = rpc_ns;
    cfg.rules = rules;
    if (!rpc_dist_spec.empty() && !(cfg.rpc_dist = load_rpc_dist(rpc_dist_spec))) return 2;
    if (poll_us) cfg.log_poll_ns = poll_us * 1000;
    if (lease_us) cfg.lease_ns = lease_us * 1000;
    if (li_cap >= 0) cfg.li_cap_factor = li_cap;
    if (heat_dedup >= 0) cfg.heat_dedup = heat_dedup != 0;
    if (heat_window_polls > 0) cfg.heat_window_polls = (uint32_t)heat_window_polls;
    cfg.shard_direct = shard_direct;
    if (ro_peers && !(perf && procs)) { fprintf(stderr, "--ro-peers 1 needs --perf --procs (one process per host)\n"); return 2; }
    cfg.ro_peers = ro_peers;
    cfg.mono = g_mono_arg == 1;
    if (mono_fd_ns) cfg.mono_fd_ns = mono_fd_ns;   // a real-time bound (backend write -> cut): not scaled
    cfg.xtra_dir_slots = xtra_dir_slots;
    cfg.xtra_own_freshness = xtra_own;
    cfg.xtra_dir_replicas = xtra_rep;
    cfg.xtra_hk_ns = xtra_hk_ns;
    cfg.li16 = li16;
    cfg.hit_clock = hit_clock;
    if (obj_read >= 0) cfg.obj_read_mode = obj_read;
    cfg.diag_lock_timing = diag_lock;
    cfg.diag_put_timing = diag_put;
    cfg.li_clock_inq = li_clock_inq;
    if (exec_slack_ns >= 0) cfg.exec_timer_slack_ns = (uint64_t)exec_slack_ns;
    if (charge_model >= 0) cfg.charge_model = charge_model;
    if (exec_lanes > 0) cfg.exec_lanes = exec_lanes;
    if (put_async >= 0) cfg.put_async = put_async;
    if (put_idle_ns >= 0) cfg.put_idle_ns = (uint64_t)put_idle_ns;
    if (handler_idle_us >= 0) cfg.handler_idle_us = (uint32_t)handler_idle_us;
    if (put_key_after >= 0) cfg.put_key_after_publish = put_key_after == 1;
    if (gen_bits) cfg.gen_bits = gen_bits;
    if (sweep_set) cfg.log_sweep_slots = sweep_slots;
    // protocol times are given in trace time; the replay runs time_scale times faster than the trace
    g_time_scale = time_scale;
    cfg.log_poll_ns = (uint64_t)((double)cfg.log_poll_ns / time_scale);
    cfg.lease_ns = (uint64_t)((double)cfg.lease_ns / time_scale);
    if (cfg.lease_tick_ns) cfg.lease_tick_ns = (uint64_t)((double)cfg.lease_tick_ns / time_scale);
    if (cfg.log_wait_ns) cfg.log_wait_ns = (uint64_t)((double)cfg.log_wait_ns / time_scale);
    if (cfg.log_interval_ns) cfg.log_interval_ns = (uint64_t)((double)cfg.log_interval_ns / time_scale);
    g_committed = rules.put_cut_at_publish && !rules.put_wait && rules.log && !rules.lease && !rules.revoke &&
                  scheme != Scheme::PRIV && scheme != Scheme::SHARD && !(xtra_own && scheme == Scheme::XTRA);
    if (g_committed) {
        ck.set_stale_bound(cfg.log_poll_ns, cfg.log_poll_ns / 50);
        cfg.on_cut = [&ck](uint64_t k, uint32_t v, uint64_t c) { ck.on_cut(k, v, c); };
        fprintf(stderr, "# replay: committed ack: checker counts bounded staleness from the cut (P = %lu ns), raw misses are info\n",
                (unsigned long)cfg.log_poll_ns);
    }
    if (g_open_rate > 0 && (pace_trace || !perf)) { fprintf(stderr, "--open-rate needs --perf and excludes --pace\n"); return 2; }
    if (pace_trace && binf.empty()) { fprintf(stderr, "--pace trace needs --bin (trace timestamps)\n"); return 2; }
    if (shard_direct && (procs || scheme != Scheme::SHARD)) { fprintf(stderr, "--shard-direct: scheme shard, threads mode only\n"); return 2; }
    cfg.heat_batch = heat;
    cfg.charge_value_bytes_only = sim_acct;
    uint64_t ai = 1;
    while (ai < 4 * (nkeys / hosts + 1)) ai <<= 1;
    cfg.ai_slots = ai;
    // local index per host: sized from the cap, not from the trace's keys: slots = cap x (objects the cache holds) / load
    // factor 0.5, the objects estimated as M / the mean slab bytes of the run's distinct keys (lines x 64 B). Without a
    // cap (--li-cap 0) every key may be held: 4 x keys.
    uint64_t li = 1, li_want;
    double est_objs = 0;
    {
        uint64_t lines = 0, n = 0;
        for (uint64_t k = 1; k <= nkeys; k++)
            if (seen[k]) { lines += obj_lines(size[k]); n++; }
        double mean = n ? 64.0 * (double)lines / (double)n : 64.0;
        est_objs = (double)M / mean;
        li_want = cfg.li_cap_factor > 0 ? (uint64_t)(cfg.li_cap_factor * est_objs / 0.5) : 4 * (nkeys + 1);
        li_want = std::min<uint64_t>(std::max<uint64_t>(li_want, 4096), 4 * (nkeys + 1));
    }
    if (li_slots_set) li_want = li_slots_set;
    while (li < li_want) li <<= 1;
    cfg.li_slots = li;
    // generation-wrap sweep slots per poll: unless --sweep-slots, the smallest count (from the Config default up,
    // doubling) whose pass over the local index keeps the startup margin check (rackcache.h log_sweep_slots) >= 10, i.e.
    // no warning
    const char *sweep_how = sweep_set ? "set" : "default";
    if (!sweep_set && cfg.rules.log && cfg.log_max_alloc_per_s > 0) {
        auto margin = [&](uint64_t s) {
            uint64_t iv = std::max<uint64_t>(cfg.log_interval_ns ? cfg.log_interval_ns : cfg.log_poll_ns / 2, 1000), se = s;
            if (cfg.log_sweep_ref_ns && iv < cfg.log_sweep_ref_ns)
                se = std::max<uint64_t>(16, (uint64_t)((double)s * (double)iv / (double)cfg.log_sweep_ref_ns));
            double pass = (double)((cfg.li_slots + se - 1) / se) * (double)iv / 1e9;
            return std::ldexp(1.0, cfg.gen_bits - 1) / (cfg.log_max_alloc_per_s * pass);
        };
        uint64_t s = cfg.log_sweep_slots ? cfg.log_sweep_slots : 16;
        while (margin(s) < 10 && s < cfg.li_slots) s *= 2;
        if (s != cfg.log_sweep_slots) {
            uint64_t lo = s / 2 + 1, hi = s;   // smallest in (s/2, s] with margin >= 10
            while (lo < hi) { uint64_t mid = (lo + hi) / 2; if (margin(mid) >= 10) hi = mid; else lo = mid + 1; }
            cfg.log_sweep_slots = std::max<uint64_t>(hi, cfg.log_sweep_slots);
            sweep_how = "auto";
        }
    }
    fprintf(stderr, "# replay: li_slots=%lu (%s; cap %g x ~%.0f objects / 0.5) log_sweep_slots=%zu (%s)\n",
            (unsigned long)cfg.li_slots, li_slots_set ? "set" : "auto", cfg.li_cap_factor, est_objs,
            cfg.log_sweep_slots, sweep_how);
    // the AI table is charged to the shared-area schemes (Impl::budget_eff subtracts it again); PRIV / SHARD have no AI
    // table and compare their private caches against budget_bytes itself, so they get exactly M / hosts (no AI-table
    // bytes as extra capacity)
    const bool private_caches = scheme == Scheme::PRIV || scheme == Scheme::SHARD;
    // CHARGE_K11B: the AI entry is in every object's charge (16 B), so the table is not charged as a whole
    cfg.budget_bytes = per_owner + (sim_acct || private_caches || cfg.charge_model == CHARGE_K11B ? 0 : ai * 16);
    // --accounting pool: only the shared CXL pool is charged to M. The local-index entries live in host DRAM and are not
    // charged (li_entry_bytes is used for charging only); per-host copies (PRIV) get the same host DRAM on top of M / hosts
    // (one 16 B entry per pool object at cap 1 x, pool objects ~ M / the prefix's mean object size). Otherwise as
    // --accounting full.
    uint64_t priv_extra = 0;
    if (pool_acct) {
        cfg.li_entry_bytes = 0;
        if (scheme == Scheme::PRIV && ndistinct) priv_extra = (uint64_t)(16.0 * (double)M / ((double)total / ndistinct));
        cfg.budget_bytes += priv_extra;
    }
    cfg.slab_bytes = (size_t)((double)per_owner * slab_factor) + (size_t)(slab_pad_mb * (1 << 20));
    cfg.ring_lines = 1 << 14;
    cfg.bind_hosts = bind;
    cfg.shm_node = shm_node;
    if (perf) {
        g_tsc_ghz = calibrate_tsc();
        g_held.assign(hosts, std::vector<uint8_t>(nkeys + 1, 0));
        uint64_t warm = (uint64_t)(warm_frac * nrec);
        if (T.B && nrec) g_trace_span_s = (double)(T.ts(nrec - 1) - T.ts(std::min(warm, nrec - 1)));
        static std::atomic<uint64_t> pace_start{0};
        if (pace_trace) { g_pace.k = time_scale; g_pace.ts0 = T.ts(0); g_pace.start = &pace_start; }
        if (g_open_rate > 0) g_pace.start = &pace_start;   // --open-rate: the same common start
        if (pace_spread) {   // even spacing of each second's records within that second
            g_spread.assign(nrec, 0.0f);
            for (uint64_t i = 0; i < nrec;) {
                uint64_t j = i;
                while (j < nrec && T.ts(j) == T.ts(i)) j++;
                for (uint64_t x = i; x < j; x++) g_spread[x] = (float)(x - i) / (float)(j - i);
                i = j;
            }
        }
        char head[2048];
        snprintf(head, sizeof head, "# replay --perf: trace=%s records=%lu warm=%lu pct=%g M=%lu per_owner=%lu heat=%d "
                 "accounting=%s seed=%lu delay_ns=%lu rpc_ns=%lu mode=%s tsc_ghz=%.4f skip=%lu routing=%s threads_per_host=%d "
                 "rpc_dist=%s slab_bytes=%zu backend_get_mean_ns=%.0f backend_put_mean_ns=%.0f open_rate=%.0f put_pool_away=%d put_inflight=%d no_puts=%d",
                 tf.c_str(), (unsigned long)nrec, (unsigned long)warm, pct, (unsigned long)M, (unsigned long)per_owner,
                 heat, sim_acct ? "sim" : pool_acct ? "pool" : "full", (unsigned long)seed, (unsigned long)delay_ns, (unsigned long)rpc_ns,
                 procs ? "procs" : "threads", g_tsc_ghz, (unsigned long)skip, routing.c_str(), nthr,
                 rpc_dist_spec.empty() ? "-" : rpc_dist_spec.c_str(), cfg.slab_bytes,
                 g_get_dist ? g_get_dist->mean : (double)delay_ns,
                 g_put_dist ? g_put_dist->mean : (double)(write_delay_ns == ~0ULL ? delay_ns : write_delay_ns),
                 g_open_rate, (int)g_put_pool_away, g_put_inflight, (int)g_no_puts);
        if (g_miss_diag && procs) { fprintf(stderr, "--miss-diag: threads mode only (no --procs)\n"); return 2; }
        if (!procs) {
            cfg.on_publish = [&](uint64_t k, uint32_t v) { ck.on_publish(k, v); };
            Cache cache(cfg, &be);
            printf("%s\n# config: %s\n", head, cache.describe().c_str());
            fflush(stdout);
            cache.start();
            cache.wait_writable();
            if (g_miss_diag) {   // per-key state for the miss classes, and each key's first GET (per host for PRIV)
                g_dc = &cache;
                g_dpriv = scheme == Scheme::PRIV;
                g_dshard = scheme == Scheme::SHARD;
                g_dN = nkeys;
                size_t nh = (size_t)hosts * (nkeys + 1);
                g_inflight_host.reset(new std::atomic<uint16_t>[nh]());
                g_inflight_key.reset(new std::atomic<uint16_t>[nkeys + 1]());
                g_fillstate.reset(new std::atomic<uint8_t>[g_dpriv ? nh : nkeys + 1]());
                g_first_get.assign(g_dpriv ? nh : nkeys + 1, UINT32_MAX);
                uint64_t rng = seed;
                for (uint64_t i = 0; i < nrec; i++) {
                    int hh = route(T, i, rng, hosts);
                    if (T.op(i) != 2) continue;
                    uint32_t &f = g_first_get[dkey(hh, T.key(i))];
                    if (f == UINT32_MAX) f = (uint32_t)i;
                }
                set_hook(diag_hook);
            }
            std::vector<HostSum> hs(hosts);
            std::vector<std::atomic<uint32_t>> vver(nkeys + 1), vseen(nkeys + 1);
            std::vector<std::atomic<uint64_t>> vret(nkeys + 1);
            Vis V{vver.data(), vseen.data(), vret.data()};
            std::vector<std::thread> th;
            for (int h = 0; h < hosts; h++)
                th.emplace_back([&, h] { run_host_threads(cache, ck, h, hosts, T, nrec, warm, seed, nthr, bind, &hs[h], V); });
            pace_start.store(now_ns(), std::memory_order_release);   // --pace trace: common start
            for (auto &t : th) t.join();
            cache.wait_idle();
            cache.stop();
            for (int h = 0; h < hosts; h++) hs[h].busy_s = cache.stats(h).handler_busy_cycles / g_tsc_ghz / 1e9;
            print_perf(hosts, hs, "perf", ck, pct, scheme, [&](int h) -> Stats & { return cache.stats(h); }, cfg.log_lines);
            print_errs(ck.first_errors());
            return ck_errors(ck) != 0 || (g_mono_arg == 1 && ck.mono_errors());
        }
        // ---- procs: one process per host; shared area and simulator state in POSIX shm
        struct Shared {
            std::atomic<int> ready{0}, done{0};
            std::atomic<uint64_t> pace_start{0};
            CheckCounters cc;
            Stats stats[16];
            HostSum sums[16];
        };
        if (hosts > 16) usage();
        std::string tag = "/rcreplay-" + std::to_string(getpid());
        cfg.shm_name = tag + "-cache";
        std::string simname = tag + "-sim";
        Cache::shm_create(cfg);
        size_t sim_bytes = sizeof(Shared) + 4 * (nkeys + 1) * sizeof(std::atomic<uint32_t>) + (nkeys + 1) * sizeof(std::atomic<uint64_t>);
        const size_t off_stale = (sim_bytes + 63) / 64 * 64;   // committed: the checker's cut ring and floors (shared)
        if (g_committed) sim_bytes = off_stale + (nkeys + 1) * Checker::CUT_RING * sizeof(std::atomic<uint64_t>) + (nkeys + 1) * sizeof(std::atomic<uint32_t>);
        int sfd = shm_open(simname.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (sfd < 0 || ftruncate(sfd, (off_t)sim_bytes)) { perror("sim shm"); return 1; }
        char *sp = (char *)mmap(nullptr, sim_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
        close(sfd);
        Shared *S = new (sp) Shared;
        auto *ver = (std::atomic<uint32_t> *)(sp + sizeof(Shared));
        auto *done = ver + (nkeys + 1);
        for (uint64_t i = 0; i <= nkeys; i++) { new (&ver[i]) std::atomic<uint32_t>(0); new (&done[i]) std::atomic<uint32_t>(0); }
        // visibility bookkeeping (shared by the host processes): ver, seen (u32), then ret (u64, 8-aligned)
        auto *vver = done + (nkeys + 1), *vseen = vver + (nkeys + 1);
        auto *vret = (std::atomic<uint64_t> *)(vseen + (nkeys + 1) + ((nkeys + 1) & 1));
        for (uint64_t i = 0; i <= nkeys; i++) {
            new (&vver[i]) std::atomic<uint32_t>(0); new (&vseen[i]) std::atomic<uint32_t>(0); new (&vret[i]) std::atomic<uint64_t>(0);
        }
        Vis V{vver, vseen, vret};
        std::atomic<uint64_t> *scuts = nullptr;
        std::atomic<uint32_t> *sfloor = nullptr;
        if (g_committed) {
            scuts = (std::atomic<uint64_t> *)(sp + off_stale);
            sfloor = (std::atomic<uint32_t> *)(scuts + (nkeys + 1) * Checker::CUT_RING);
            for (uint64_t i = 0; i < (nkeys + 1) * Checker::CUT_RING; i++) new (&scuts[i]) std::atomic<uint64_t>(0);
            for (uint64_t i = 0; i <= nkeys; i++) new (&sfloor[i]) std::atomic<uint32_t>(0);
        }
        if (pace_trace || g_open_rate > 0) g_pace.start = &S->pace_start;   // children inherit g_pace; the parent sets the start below
        cfg.shared_stats = S->stats;
        printf("%s\n", head);
        fflush(stdout);
        std::vector<pid_t> kids;
        for (int h = 0; h < hosts; h++) {
            pid_t pid = fork();
            if (pid == 0) {
                if (bind && numa_available() >= 0) {
                    numa_run_on_node(h % (numa_max_node() + 1));
                    numa_set_preferred(h % (numa_max_node() + 1));
                }
                DistBackend b2(nkeys, delay_ns, ver);
                if (write_delay_ns != ~0ULL) b2.set_put_delay(write_delay_ns);
                b2.set_sizes(size.data());
                Checker c2(nkeys, &b2, done, &S->cc, scuts, sfloor);
                if (g_mono_arg >= 0) c2.enable_mono(hosts);
                Config c = cfg;
                c.local_host = h;
                if (g_committed) {
                    c2.set_stale_bound(cfg.log_poll_ns, cfg.log_poll_ns / 50);
                    c.on_cut = [&](uint64_t k, uint32_t v, uint64_t ct) { c2.on_cut(k, v, ct); };
                }
                const int llc_pfd = g_llc && h == g_llc_host ? llc_open(true) : -1;   // --llc-count: the whole process
                c.on_publish = [&](uint64_t k, uint32_t v) { c2.on_publish(k, v); };
                {
                    Cache cache(c, &b2);
                    if (h == 0) { printf("# config: %s\n", cache.describe().c_str()); fflush(stdout); }
                    cache.start();
                    cache.wait_writable();
                    S->ready++;
                    while (S->ready.load() < hosts) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    run_host_threads(cache, c2, h, hosts, T, nrec, warm, seed, nthr, false, &S->sums[h], V);
                    S->done++;
                    while (S->done.load() < hosts) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    cache.stop();
                    rusage ru{};   // the whole host process (clients, PUT pool, the owner's service threads)
                    getrusage(RUSAGE_SELF, &ru);
                    S->sums[h].cpu_proc_ns = (uint64_t)(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000000000ULL +
                                             (uint64_t)(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1000ULL;
                    if (llc_pfd >= 0) { uint64_t v = 0; if (read(llc_pfd, &v, sizeof v) == (ssize_t)sizeof v) S->sums[h].llc_proc = v; close(llc_pfd); }
                    print_errs(c2.first_errors(), h);
                    fflush(stdout);
                }
                _exit(0);
            }
            kids.push_back(pid);
        }
        int bad = 0;
        while (S->ready.load() < hosts) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        S->pace_start.store(now_ns(), std::memory_order_release);   // --pace trace: common start for all hosts
        for (pid_t k : kids) { int st = 0; waitpid(k, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) bad++; }
        std::vector<HostSum> hs(S->sums, S->sums + hosts);
        for (int h = 0; h < hosts; h++) hs[h].busy_s = S->stats[h].handler_busy_cycles / g_tsc_ghz / 1e9;
        SimBackend dummy(1, 0);
        Checker view(nkeys, &dummy, done, &S->cc);
        print_perf(hosts, hs, "perf-procs", view, pct, scheme, [&](int h) -> Stats & { return S->stats[h]; }, cfg.log_lines);
        Cache::shm_unlink(cfg.shm_name);
        shm_unlink(simname.c_str());
        printf("# bad_child_exits=%d\n", bad);
        return ck_errors(view) != 0 || bad != 0 || (g_mono_arg == 1 && view.mono_errors());
    }
    cfg.on_publish = [&](uint64_t k, uint32_t v) { ck.on_publish(k, v); };
    Cache cache(cfg, &be);
    printf("# replay: trace=%s records=%lu keys=%lu total_bytes=%lu pct=%g M=%lu per_owner=%lu heat=%d accounting=%s "
           "seed=%lu delay_ns=%lu\n# config: %s\n",
           tf.c_str(), (unsigned long)nrec, (unsigned long)nkeys, (unsigned long)total, pct, (unsigned long)M,
           (unsigned long)per_owner, heat, sim_acct ? "sim" : pool_acct ? "pool" : "full", (unsigned long)seed, (unsigned long)delay_ns,
           cache.describe().c_str());
    fflush(stdout);
    cache.start();
    cache.wait_writable();
    std::vector<std::unique_ptr<Client>> cl;
    for (int h = 0; h < hosts; h++) cl.emplace_back(new Client(&cache, h));
    // ---- replay
    uint64_t rng = seed, gets = 0, hits = 0, fast = 0, slow = 0, ghost = 0, writes = 0, dels = 0, other = 0;
    uint64_t ttl_expired = 0, ttl_miss = 0;   // --ttl-side
    std::vector<char> buf(MAX_OUT);
    std::vector<uint64_t> nreq(hosts, 0);
    uint64_t t0 = now_ns();
    auto line = [&](uint64_t i) {
        double el = (now_ns() - t0) / 1e9;
        printf("i=%lu (%.1f%%) gets=%lu hit=%.6f fast=%lu slow=%lu ghost=%lu hit_excl_ghost=%.6f writes=%lu "
               "elapsed=%.0fs rate=%.2fM/s | %s\n",
               (unsigned long)i, 100.0 * i / nrec, (unsigned long)gets, gets ? (double)hits / gets : 0, (unsigned long)fast,
               (unsigned long)slow, (unsigned long)ghost, gets ? (double)(hits - ghost) / gets : 0, (unsigned long)writes,
               el, i / el / 1e6, ck.summary().c_str());
        fflush(stdout);
    };
    for (uint64_t i = 0; i < nrec; i++) {
        struct { uint64_t key; } r{T.key(i)};
        int h = route(T, i, rng, hosts);   // random stream drawn for every record, as in the simulators
        int t = T.op(i);
        if (t == 2) {
            gets++;
            bool expired = false;
            if (g_ttl_side && ttl_expire_before(*cl[h], T, i, r.key)) {   // the EXPIRE completes (and its messages) first
                expired = true;
                ttl_expired++;
                cache.wait_idle();
                if (heat > 0 && ++nreq[h] % heat == 0) cache.wait_idle();   // the client's request count (heat batches)
            }
            uint32_t req = ck.begin(r.key);
            const uint32_t sfl = ck.stale_begin(r.key);
            const uint32_t mfl = ck.mono_begin(h, r.key);
            Kind kind;
            uint32_t vlen = cl[h]->get(r.key, buf.data(), &kind);
            ck.verify(r.key, req, buf.data(), vlen, h, kind);
            if (sfl) { uint32_t gv; memcpy(&gv, buf.data() + 8, 4); ck.verify_stale(r.key, sfl, gv, h, kind); }
            if (ck.mono_on()) { uint32_t gv; memcpy(&gv, buf.data() + 8, 4); ck.verify_mono(h, r.key, mfl, gv, kind); }
            if (expired && kind == MISS) ttl_miss++;
            if (g_ttl_side && kind == MISS && vlen) ttl_install(T, i, r.key, false);
            if (kind == MISS) {
                cache.wait_idle();
            } else {
                hits++;
                (kind == FAST ? fast : slow)++;
                if (scheme != Scheme::PRIV && scheme != Scheme::SHARD && !cache.owner_cached(r.key)) ghost++;
            }
        } else if (t == 1 || t == 3) {
            writes++;
            if (g_ttl_side) ttl_install(T, i, r.key, t == 3);
            ck.put_done(r.key, cl[h]->put(r.key));
            cache.wait_idle();
        } else if (t == 4) {
            dels++;   // without --deletes 1: skipped (cluster17 has none)
            if (g_deletes) {   // a DELETE
                if (g_ttl_side) g_ttl_exp[r.key].store(0, std::memory_order_relaxed);
                ck.put_done(r.key, cl[h]->del(r.key));
                cache.wait_idle();
            }
        } else {
            other++;
        }
        // the client flushed its heat batch at this request: let the owners apply it before going on (serial)
        if (heat > 0 && (t == 1 || t == 2 || t == 3 || (t == 4 && g_deletes)) && ++nreq[h] % heat == 0) cache.wait_idle();
        if ((i + 1) % report_every == 0) line(i + 1);
    }
    line(nrec);
    for (auto &c : cl) { c->flush_heat(); c->flush_stats(); }
    cache.wait_idle();
    cache.stop();
    double hr = gets ? (double)hits / gets : 0, hrx = gets ? (double)(hits - ghost) / gets : 0;
    if (g_deletes) printf("# deletes=%lu other=%lu\n", (unsigned long)dels, (unsigned long)other);
    else printf("# deletes_ignored=%lu other=%lu\n", (unsigned long)dels, (unsigned long)other);
    for (int o = 0; o < hosts; o++) {
        Cache::Budget b = cache.owner_budget(o);
        Stats &x = cache.stats(o);
        printf("# host %d: objects=%lu object_bytes=%lu budget=%lu evictions=%lu frag_evictions=%lu admits=%lu "
               "fill_drop[tok=%lu pend=%lu cached=%lu space=%lu] ring_drop=%lu fast_fail=%lu slow_fail=%lu ai_absent=%lu "
               "tomb_refuse=%lu heat_ok=%lu\n", o, (unsigned long)b.nobj, (unsigned long)b.objects, (unsigned long)b.limit,
               (unsigned long)x.evictions.load(), (unsigned long)x.frag_evictions.load(), (unsigned long)x.admits.load(),
               (unsigned long)x.fill_drop_tok.load(), (unsigned long)x.fill_drop_pending.load(),
               (unsigned long)x.fill_drop_cached.load(), (unsigned long)x.fill_drop_space.load(),
               (unsigned long)x.drop_ring_full.load(), (unsigned long)x.fast_fail.load(), (unsigned long)x.slow_fail.load(),
               (unsigned long)x.ai_absent.load(), (unsigned long)x.tomb_refuse.load(), (unsigned long)x.heat_applied.load());
    }
    printf("RESULT pct=%g heat=%d accounting=%s gets=%lu hit_ratio=%.6f hit_ratio_excl_ghost=%.6f", pct, heat,
           sim_acct ? "sim" : pool_acct ? "pool" : "full", (unsigned long)gets, hr, hrx);
    if (expect >= 0) printf(" sim=%.6f diff_pp=%+.3f diff_excl_ghost_pp=%+.3f", expect, 100 * (hr - expect), 100 * (hrx - expect));
    printf(" errors=%lu%s\n", (unsigned long)ck_errors(ck), g_committed ? (" committed=1 raw_misses=" + std::to_string(ck.p2_errors()) +
                                                                       " stale_errors=" + std::to_string(ck.stale_errors())).c_str() : "");
    if (g_deletes || g_ttl_side) {   // only with --deletes 1 / --ttl-side
        uint64_t ga = 0, od = 0, oe = 0, xd = 0;
        for (int h = 0; h < hosts; h++) {
            ga += cache.stats(h).get_absent; od += cache.stats(h).deletes; oe += cache.stats(h).expires; xd += cache.stats(h).expire_dropped;
        }
        printf("WRITES deletes=%lu ttl_side=%s ttl_expired=%lu ttl_expired_per_get=%.6f ttl_expired_then_miss=%lu owner_deletes=%lu "
               "owner_expires=%lu expire_dropped=%lu get_absent=%lu\n", (unsigned long)(g_deletes ? dels : 0),
               g_ttl_side ? g_ttl_side_f.c_str() : "-", (unsigned long)ttl_expired, gets ? (double)ttl_expired / gets : 0.0,
               (unsigned long)ttl_miss, (unsigned long)od, (unsigned long)oe, (unsigned long)xd, (unsigned long)ga);
    }
    print_mono(hosts, [&](int h) -> Stats & { return cache.stats(h); }, ck);
    print_errs(ck.first_errors());
    return ck_errors(ck) != 0 || (g_mono_arg == 1 && ck.mono_errors());
}
