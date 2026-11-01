// Krater prototype implementation, with the compared designs. See rackcache.h for the protocol rules and the shared formats.
#include "rackcache.h"
#include "../../bench/rpc/rpc_dist.h"

#include <immintrin.h>
#include <numa.h>
#include <numaif.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace rc {

#define barrier() __asm__ volatile("" ::: "memory")

static inline uint32_t vld32(const void *p) { return *(const volatile uint32_t *)p; }
static inline uint64_t vld64(const void *p) { return *(const volatile uint64_t *)p; }
static inline void vst32(void *p, uint32_t v) { *(volatile uint32_t *)p = v; }
static inline void vst64(void *p, uint64_t v) { *(volatile uint64_t *)p = v; }
// Shared-memory access counts per operation (the paper's §4.4 table): this thread's flushes, CXL line
// writes (each line written back with clwb) and messages are counted here as they happen; CXL line reads are counted at
// the read sites (one per 64 B line read, however many loads it takes). Control-channel ring traffic is not counted
// (send() restores the counters; a message counts as one message). Operations take a snapshot and add the difference
// to their bucket (ShmOp; Client: per-client counters merged as the others).
struct ShmCount { uint64_t flush = 0, rd = 0, wr = 0, msg = 0; };
static thread_local ShmCount tl_shm;
static inline void flush_line(const void *p) { _mm_clflushopt((void *)p); tl_shm.flush++; }
static inline void wb_line(void *p) { _mm_clwb(p); tl_shm.wr++; }
static inline void diag_add(Stats &S, int k, uint64_t cyc) {   // diag_lock_timing histograms (rackcache.h Stats)
    S.diag_hist[k][cyc ? std::min(47, 64 - __builtin_clzll(cyc)) : 0].fetch_add(1, std::memory_order_relaxed);
    S.diag_sum[k].fetch_add(cyc, std::memory_order_relaxed);
}
static inline void shm_add(std::atomic<uint64_t> *b, const ShmCount &s0) {   // b[5]: flush, rd, wr, msg, ops
    b[0].fetch_add(tl_shm.flush - s0.flush, std::memory_order_relaxed);
    b[1].fetch_add(tl_shm.rd - s0.rd, std::memory_order_relaxed);
    b[2].fetch_add(tl_shm.wr - s0.wr, std::memory_order_relaxed);
    b[3].fetch_add(tl_shm.msg - s0.msg, std::memory_order_relaxed);
    b[4].fetch_add(1, std::memory_order_relaxed);
}
static inline uint64_t mono_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
// TSC cycles per ns, measured once per process (the slow path's epoch check uses the TSC, not the epoch thread)
static double tsc_per_ns() {
    static double r = [] {
        uint64_t t0 = mono_ns(), c0 = __rdtsc();
        while (mono_ns() - t0 < 5'000'000) _mm_pause();
        return (double)(__rdtsc() - c0) / (double)(mono_ns() - t0);
    }();
    return r;
}
static inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------- hooks
static std::atomic<bool> g_hook_on{false};
static HookFn g_hook;
void set_hook(HookFn f) {
    g_hook_on.store(false);
    g_hook = std::move(f);
    g_hook_on.store((bool)g_hook);
}
const char *hook_name(Hook h) {
    static const char *n[] = {"slow.after_ai_read", "slow.after_install", "miss.after_token", "miss.before_send_fill",
                              "put.after_bump1", "put.after_acks", "put.after_backend", "owner.after_gen_store",
                              "priv.writer_before_install", "fast.after_li_read", "fast.after_load", "put.after_step1",
                              "poll.after_tails", "fill.done", "put.swap_mid"};
    return n[(int)h];
}
static inline void hook(Hook h, int host, uint64_t key) {
    if (__builtin_expect(g_hook_on.load(std::memory_order_relaxed), 0)) g_hook(h, host, key);
}

bool set_update(Rules &r, const std::string &n) {
    if (n == "log") { r.log = true; r.lease = false; r.revoke = false; }
    else if (n == "lease") { r.log = false; r.lease = true; r.revoke = false; }
    else if (n == "invalidate") { r.log = false; r.lease = false; r.revoke = true; }
    else return false;
    return true;
}
const char *update_name(const Rules &r) {
    if (r.log) return "log";
    if (r.lease && r.revoke) return "lease+invalidate";
    if (r.lease) return "lease";
    if (r.revoke) return "invalidate";
    return "none";
}

struct RpcDist { rpc_dist d; };
std::shared_ptr<const RpcDist> load_rpc_dist(const std::string &spec) {
    size_t a = spec.find(':'), b = a == std::string::npos ? a : spec.find(':', a + 1);
    if (b == std::string::npos) { fprintf(stderr, "--rpc-dist wants <dir>:<kind>:<client>\n"); return nullptr; }
    auto r = std::make_shared<RpcDist>();
    if (rpc_dist_load(&r->d, spec.substr(0, a).c_str(), spec.substr(a + 1, b - a - 1).c_str(), spec.substr(b + 1).c_str()))
        return nullptr;
    return r;
}

static const char *SCHEME_NAMES[] = {"rackcache", "cxl_uc", "ver", "ver_noflush", "gstate_cxl", "gstate_coh", "priv", "shard", "xtra"};
const char *scheme_name(Scheme s) { return SCHEME_NAMES[(int)s]; }
static const char *OBJ_READ_NAMES[] = {"copy", "prefetch", "mstore", "prefetch+mstore"};
int obj_read_mode_parse(const std::string &s) {
    if (s == "both") return 3;
    for (int i = 0; i < 4; i++)
        if (s == OBJ_READ_NAMES[i]) return i;
    return -1;
}
const char *obj_read_mode_name(int m) { return m >= 0 && m < 4 ? OBJ_READ_NAMES[m] : "?"; }
// masked-store copies need AVX-512 BW, at compile time (-march) and on the CPU running the binary
static bool mstore_available() {
#if defined(__AVX512BW__)
    static const bool ok = __builtin_cpu_supports("avx512bw");
    return ok;
#else
    return false;
#endif
}
bool scheme_parse(const std::string &n, Scheme *out) {
    if (n == "priv_host") { *out = Scheme::PRIV; return true; }   // PRIV is one private copy per host, shared by its threads
    for (int i = 0; i < (int)(sizeof SCHEME_NAMES / sizeof SCHEME_NAMES[0]); i++)
        if (n == SCHEME_NAMES[i]) { *out = (Scheme)i; return true; }
    return false;
}

// ---------------------------------------------------------------- shared-area formats
// AI entry: one 16 B item, written with one 16 B store and read with one 16 B load (never torn).
//   w0 = slab offset in lines (low 28 bits) | size class << 28;  gen;  kg;  fp = 32-bit key fingerprint.
// empty = all zero; deleted = gen 0 with w0 = DELETED_W0. Readers stop at empty, skip deleted.
enum : uint8_t { AI_EMPTY = 0, AI_FULL = 1, AI_DELETED = 2 };
struct alignas(16) AiEnt { uint32_t w0, gen, kg, fp; };
static_assert(sizeof(AiEnt) == 16, "AiEnt");
constexpr uint32_t DELETED_W0 = 0xFFFFFFFFu, OFF_MASK = (1u << 28) - 1;
static inline uint32_t key_fp(uint64_t k) { return (uint32_t)(mix64(k ^ 0x27d4eb2f165667c5ULL) >> 32); }
static inline void store16(AiEnt *dst, const AiEnt &v) {
    __m128i x = _mm_loadu_si128((const __m128i *)&v);
    __asm__ volatile("vmovdqa %1, %0" : "=m"(*dst) : "x"(x));
}
// the same 16 B single-store / single-load for any 16 B item (the log's tail line)
static inline void store16v(void *dst, const void *v) {
    __m128i x = _mm_loadu_si128((const __m128i *)v);
    __asm__ volatile("vmovdqa %1, %0" : "=m"(*(char(*)[16])dst) : "x"(x));
}
static inline void load16v(const void *src, void *v) {
    __m128i x;
    __asm__ volatile("vmovdqa %1, %0" : "=x"(x) : "m"(*(const char(*)[16])src));
    _mm_storeu_si128((__m128i *)v, x);
}
static inline AiEnt load16(const AiEnt *src) {
    __m128i x;
    __asm__ volatile("vmovdqa %1, %0" : "=x"(x) : "m"(*src));
    AiEnt v;
    _mm_storeu_si128((__m128i *)&v, x);
    return v;
}
#if defined(__AVX512F__)
// one aligned 64 B load of a line (single instance, H5), spilled to a local buffer
static inline void load64(const char *src, char *dst) {
    __m512i x;
    __asm__ volatile("vmovdqa64 %1, %0" : "=v"(x) : "m"(*(const char(*)[64])src));
    _mm512_storeu_si512(dst, x);
}
static inline __m512i vload64(const char *src) {   // one aligned 64 B load into a register (as load64)
    __m512i x;
    __asm__ volatile("vmovdqa64 %1, %0" : "=v"(x) : "m"(*(const char(*)[64])src));
    return x;
}
#else
// FUNCTIONAL-TEST build for CPUs without AVX-512 (e.g., AMD Zen 3): a line is read with two 32 B loads,
// so H5 (one 64 B load per line) does not hold -- fixed-point tests only, never a measurement or a torn-read argument
static inline void load64(const char *src, char *dst) {
    __m256i a, b;
    __asm__ volatile("vmovdqa %1, %0" : "=x"(a) : "m"(*(const char(*)[32])src));
    __asm__ volatile("vmovdqa %1, %0" : "=x"(b) : "m"(*(const char(*)[32])(src + 32)));
    _mm256_storeu_si256((__m256i *)dst, a);
    _mm256_storeu_si256((__m256i *)(dst + 32), b);
}
#endif

enum MsgType : uint32_t { M_PAD = 0, M_TOKREQ, M_TOK, M_FILL, M_REVOKE, M_ACK, M_PUTREQ, M_PUTDONE, M_HEAT,
                          M_PINV, M_PDONE,     // PRIV: PUT round 1 (invalidate, ack) and round 2 (done)
                          M_SGET, M_SVAL, M_SMISS, M_SFILL,     // SHARD: GET RPC, value / miss reply, fill
                          M_RECL, M_RACK };                     // delayed reuse: drop + flush a slot, ack
struct alignas(64) MsgHdr {
    uint32_t type, nlines;   // nlines includes this header line
    uint64_t key;
    uint32_t a, b, c, d;     // TOKREQ a=reqid | TOK a=reqid b=kg | FILL a=tok b=vlen c=ver | REVOKE b=kg
    uint64_t e;              // REVOKE/ACK e=putid | PUTREQ a=reqid c=WriteOp d=seq | PUTDONE a=reqid b=ver d=seq | HEAT a=count
    uint8_t pad[64 - 40];
};
static_assert(sizeof(MsgHdr) == 64, "MsgHdr");
struct HeatEnt { uint64_t off; uint32_t gen, rsv; };   // 4 per line

// Owner log. In the owner's region: one tail line, then log_lines record lines.
//   tail line: one 16 B item {te = tail (records ever appended in this epoch, 40 bits) | epoch << 40, count = the owner's
//              allocation counter (true count; generations are count mod 2^gen_bits)}, written with one 16 B store
//              after the record line is written back, read with one 16 B load after a flush.
//              The second 16 B item of the tail line holds the owner's object count (for the local-index cap), every ms.
//   record line, log_rec8 (default): [header 8 B = lap + 1 (bits 0-55) | refresh-start map (bits 56-62)]
//              then 7 slots of 8 B. The tail counts slots; slot i is in line (i / 7) % log_lines, position i % 7, lap
//              (i / 7) / log_lines. A delete record is one slot holding the key (keys are never 0); a refresh record is
//              two slots of one line, the key then (w0 << 32 | generation), and sets its start position's bit in the
//              map; a refresh that would start in the last slot pads it with 0 and starts the next line. Deletes: 7 per
//              line instead of 3 with !log_rec8.
//   record line, !log_rec8: [lap + 1 (8 B) | 8 B pad | 3 records of 16 B]; the tail counts records; record i is in
//              line (i / 3) % log_lines, slot i % 3, lap (i / 3) / log_lines.
//   Either way a reader that finds another lap in the header was lapped (records lost).
//   record: {key (8 B; 64-bit keys are their own fingerprint), w0 = slot in lines | size class << 28, generation}; a
//              delete record (PUT, at its cut) has w0 = DEL_W0; a refresh record (after every publish) names the new
//              object.
struct alignas(16) TailEnt { uint64_t te, count; };
static_assert(sizeof(TailEnt) == 16, "TailEnt");
struct alignas(16) LogRec { uint64_t key; uint32_t w0, gen; };
static_assert(sizeof(LogRec) == 16, "LogRec");
constexpr uint64_t RPL = 3, SPL = 7, TAIL_MASK = (1ULL << 40) - 1, LAP_MASK = (1ULL << 56) - 1;
constexpr uint32_t DEL_W0 = 0xFFFFFFFFu;
constexpr size_t REC_BUCKETS = 4096;   // per (reader host, owner): seq of the last applied record per key bucket

// size classes in lines
static const uint32_t CLS[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192};
constexpr int NCLS = sizeof(CLS) / sizeof(CLS[0]);
static_assert(NCLS <= 16, "size class must fit in 4 bits");
static inline int class_of(uint32_t nl) {
    for (int i = 0; i < NCLS; i++) if (CLS[i] >= nl) return i;
    return -1;
}

// Config::charge_model = CHARGE_K11B: slab size classes (64 B, then x 1.25 rounded up to a multiple of 64 B)
uint64_t k11b_slab(uint64_t b) {
    static const std::vector<uint64_t> t = [] {
        std::vector<uint64_t> v{64};
        while (v.back() < (1u << 22)) v.push_back(std::max((v.back() * 5 / 4 + 63) / 64 * 64, v.back() + 64));
        return v;
    }();
    return *std::lower_bound(t.begin(), t.end(), b);
}
// bytes charged per object in each design's own object format (shared-pool part)
uint64_t charge_k11b(Scheme s, uint32_t v) {
    switch (s) {
    case Scheme::XTRA: return k11b_slab(v) + 24;                                  // + AI entry 16 + directory slot 8
    case Scheme::CXL_UC: return k11b_slab((uint64_t)v + 16) + 24;                  // Beluga-style: header 16, entry 24
    case Scheme::GSTATE_CXL: case Scheme::GSTATE_COH: return k11b_slab(v);         // MEGALON-style, pool only
    case Scheme::PRIV: case Scheme::SHARD: return k11b_slab(v) + 16;               // + host index entry
    default: return k11b_slab((uint64_t)v + 4 * (((uint64_t)v + 59) / 60)) + 16;  // Krater: 4 B gen per line + AI 16
    }
}

// ---------------------------------------------------------------- host-local state
struct LiEnt { uint64_t off; uint32_t gen, nl; };
struct TsEnt { uint32_t kg; uint64_t t; };   // tombstone: key generation and write time
constexpr int NSTRIPE = 256;
struct Stripe {   // serializes the writers of one key (LI updates and its tombstone)
    std::mutex mu;
    std::unordered_map<uint64_t, TsEnt> ts;      // tombstones
};

// Local index: open addressing; each slot is a seqlock, so the fast path reads it without any lock. Writers of
// one key are serialized by its stripe lock; writers of different keys that meet in a slot take the slot by
// CAS-ing its sequence number to odd. Keys 0 (empty) and ~0 (deleted) are reserved.
constexpr uint64_t LI_EMPTY = 0, LI_DEL = ~0ULL;
struct alignas(32) LiSlot {
    std::atomic<uint32_t> seq{0};
    std::atomic<uint32_t> ep{0};                   // lease: epoch e0 of the entry
    std::atomic<uint64_t> key{LI_EMPTY}, val{0};   // val = gen | off_lines << 32 | cls << 60
    std::atomic<uint8_t> ref{0};                   // local-index cap: read since the CLOCK hand last passed (refreshes do not set it)
    std::atomic<uint16_t> hwin{0};                 // heat reports: heat window (mod 2^16) this host last reported the entry in
};
static_assert(sizeof(LiSlot) == 32, "LiSlot");
static inline uint64_t li_pack(uint64_t off, uint32_t gen, int cls) {
    return (uint64_t)gen | (off / LINE) << 32 | (uint64_t)cls << 60;
}
struct LiTable {
    std::unique_ptr<LiSlot[]> s;
    uint64_t mask = 0;
    int max_probe = 128;
    std::atomic<int64_t> count{0};   // keys held (valid entries and markers)
    void init(size_t slots) { s.reset(new LiSlot[slots]); mask = slots - 1; }
    uint64_t home(uint64_t k) const { return mix64(k ^ 0x3c6ef372fe94f82bULL) & mask; }
    // lock-free read of one slot
    static void read(const LiSlot &x, uint64_t *k, uint64_t *v, uint32_t *ep = nullptr) {
        for (;;) {
            uint32_t s1 = x.seq.load(std::memory_order_acquire);
            if (s1 & 1) { _mm_pause(); continue; }
            *k = x.key.load(std::memory_order_relaxed);
            *v = x.val.load(std::memory_order_relaxed);
            if (ep) *ep = x.ep.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (x.seq.load(std::memory_order_relaxed) == s1) return;
        }
    }
    bool find(uint64_t key, uint64_t *v, uint32_t *ep = nullptr, LiSlot **slot = nullptr) const {
        uint64_t i = home(key);
        for (int n = 0; n < max_probe; n++, i = (i + 1) & mask) {
            uint64_t k, val;
            uint32_t e;
            read(s[i], &k, &val, &e);
            if (k == LI_EMPTY) return false;
            if (k == key) { *v = val; if (ep) *ep = e; if (slot) *slot = &s[i]; return true; }
        }
        return false;
    }
    static void lock(LiSlot &x) {
        for (;;) {
            uint32_t s1 = x.seq.load(std::memory_order_relaxed);
            if (!(s1 & 1) && x.seq.compare_exchange_weak(s1, s1 + 1, std::memory_order_acquire)) break;
            _mm_pause();
        }
        std::atomic_thread_fence(std::memory_order_release);
    }
    static void unlock(LiSlot &x) { x.seq.fetch_add(1, std::memory_order_release); }
    // the caller holds the key's stripe lock. Returns false if no slot within max_probe (entry not installed).
    // touch: also mark it read (local-index cap; a refresh repoints without touching)
    bool put(uint64_t key, uint64_t v, uint32_t ep = 0, bool touch = false) {
        uint64_t i = home(key), free_i = ~0ULL;
        for (int n = 0; n < max_probe; n++, i = (i + 1) & mask) {
            uint64_t k, val;
            read(s[i], &k, &val);
            if (k == key) {
                lock(s[i]);
                if (s[i].key.load(std::memory_order_relaxed) == key) {
                    s[i].val.store(v, std::memory_order_relaxed);
                    s[i].ep.store(ep, std::memory_order_relaxed);
                    if (touch) s[i].ref.store(1, std::memory_order_relaxed);
                    unlock(s[i]);
                    return true;
                }
                unlock(s[i]);
                return put(key, v, ep, touch);   // the slot changed under us: retry
            }
            if ((k == LI_DEL || k == LI_EMPTY) && free_i == ~0ULL) free_i = i;
            if (k == LI_EMPTY) break;
        }
        for (uint64_t j = free_i; free_i != ~0ULL;) {   // claim the first free slot on the chain
            lock(s[j]);
            uint64_t k = s[j].key.load(std::memory_order_relaxed);
            if (k == LI_EMPTY || k == LI_DEL) {
                s[j].val.store(v, std::memory_order_relaxed);
                s[j].ep.store(ep, std::memory_order_relaxed);
                s[j].key.store(key, std::memory_order_relaxed);
                s[j].ref.store(1, std::memory_order_relaxed);
                unlock(s[j]);
                count.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            unlock(s[j]);
            return put(key, v, ep, touch);   // another key took it: rescan
        }
        return false;
    }
    // keep_failed: mark key's entry invalid (val = FAILED_MARK: generation 0) if it still holds expect_v; stays held.
    // Caller holds the stripe lock.
    bool mark_failed(uint64_t key, uint64_t expect_v) {
        uint64_t i = home(key);
        for (int n = 0; n < max_probe; n++, i = (i + 1) & mask) {
            uint64_t k, val;
            read(s[i], &k, &val);
            if (k == LI_EMPTY) return false;
            if (k != key) continue;
            lock(s[i]);
            bool done = s[i].key.load(std::memory_order_relaxed) == key && s[i].val.load(std::memory_order_relaxed) == expect_v;
            if (done) s[i].val.store(1ULL << 63, std::memory_order_relaxed);
            unlock(s[i]);
            return done;
        }
        return false;
    }
    // log: mark key's entry as deleted by a log record (val = 0: never a valid generation); keeps the slot, so a later
    // lookup can tell "deleted by the log" from "never installed". Caller holds the stripe lock. True if marked.
    bool mark_deleted(uint64_t key) {
        uint64_t i = home(key);
        for (int n = 0; n < max_probe; n++, i = (i + 1) & mask) {
            uint64_t k, val;
            read(s[i], &k, &val);
            if (k == LI_EMPTY) return false;
            if (k != key) continue;
            if (!(uint32_t)val) return false;   // already a marker
            lock(s[i]);
            bool done = s[i].key.load(std::memory_order_relaxed) == key;
            if (done) s[i].val.store(0, std::memory_order_relaxed);
            unlock(s[i]);
            return done;
        }
        return false;
    }
    // erase key (if expect_v != 0: only if its value still equals expect_v); caller holds the stripe lock.
    // Returns true if an entry was removed.
    bool erase(uint64_t key, uint64_t expect_v = 0) {
        uint64_t i = home(key);
        for (int n = 0; n < max_probe; n++, i = (i + 1) & mask) {
            uint64_t k, val;
            read(s[i], &k, &val);
            if (k == LI_EMPTY) return false;
            if (k != key) continue;
            bool done = false;
            lock(s[i]);
            if (s[i].key.load(std::memory_order_relaxed) == key && (!expect_v || s[i].val.load(std::memory_order_relaxed) == expect_v)) {
                s[i].key.store(LI_DEL, std::memory_order_relaxed);
                done = true;
            }
            unlock(s[i]);
            if (done) count.fetch_sub(1, std::memory_order_relaxed);
            return done;
        }
        return false;
    }
};
// The local index with 16 B entries (Config::li16; log scheme only). Each slot is two words:
//   v = the entry: gen | off_lines << 32 | cls << 60 (as LiSlot::val; 0 = deleted by a log record, 1 << 63 = failed check)
//   m = seq 16 (seqlock, bits 63..48) | fp 20 (47..28: 0 empty, 1 deleted, else the key's fingerprint) | owner 5 (27..23)
//       | ep 16 (22..7: the owner's drop generation mod 2^16) | ref 1 (6: read since the CLOCK hand passed) | hwin 6 (5..0)
// No key is stored: a lookup matches fingerprint + owner. Two keys of one owner with the same fingerprint share an entry
// (the later install overwrites; records for either apply to it); a hit is still checked against the full key in the
// object's line 0 (obj_read), so reading another key's object fails the check -> slow path. Different owners never
// alias (owner field), so an entry's ep is always compared with its own owner's drop generation. ep is 16 bits: when an
// owner's drop generation reaches a multiple of 2^16 the poller clears all of that owner's entries (li16_clear_owner),
// so no entry lives across a full wrap of ep. Readers write only ref / hwin (atomic RMW on m, never the seq field).
struct alignas(16) LiSlot16 {
    std::atomic<uint64_t> m{0}, v{0};
};
static_assert(sizeof(LiSlot16) == 16, "LiSlot16");
namespace l16 {
constexpr int SEQ_SH = 48, FP_SH = 28, OWN_SH = 23, EP_SH = 7;
constexpr uint64_t SEQ_ONE = 1ULL << SEQ_SH, FP_MASK = (1ULL << 20) - 1, OWN_MASK = 31, EP_MASK = 0xFFFF, REF = 1ULL << 6,
                   HWIN_MASK = 63;
constexpr uint64_t FIELDS = (FP_MASK << FP_SH) | (OWN_MASK << OWN_SH) | (EP_MASK << EP_SH);
constexpr uint64_t FP_EMPTY = 0, FP_DEL = 1;
static inline uint64_t fp(uint64_t m) { return m >> FP_SH & FP_MASK; }
static inline int owner(uint64_t m) { return (int)(m >> OWN_SH & OWN_MASK); }
static inline uint32_t ep(uint64_t m) { return (uint32_t)(m >> EP_SH & EP_MASK); }
static inline uint64_t tag(uint64_t key, int o, uint64_t nfp) {   // fingerprint (nfp values) + owner, as placed in m
    uint64_t f = 2 + (mix64(key ^ 0x6a09e667f3bcc909ULL) >> 44) % nfp;
    return f << FP_SH | (uint64_t)o << OWN_SH;
}
constexpr uint64_t TAG_MASK = (FP_MASK << FP_SH) | (OWN_MASK << OWN_SH);
}
struct LiTable16 {
    std::unique_ptr<LiSlot16[]> s;
    uint64_t mask = 0;
    int max_probe = 128;
    std::atomic<int64_t> count{0};
    void init(size_t slots) { s.reset(new LiSlot16[slots]); mask = slots - 1; }
    uint64_t home(uint64_t k) const { return mix64(k ^ 0x3c6ef372fe94f82bULL) & mask; }
    // consistent snapshot of one slot (m's seq unchanged across the read of v; ref / hwin may move, they are hints)
    static void read(const LiSlot16 &x, uint64_t *m, uint64_t *v) {
        for (;;) {
            uint64_t m1 = x.m.load(std::memory_order_acquire);
            if (m1 >> l16::SEQ_SH & 1) { _mm_pause(); continue; }
            *v = x.v.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if ((x.m.load(std::memory_order_relaxed) ^ m1) >> l16::SEQ_SH == 0) { *m = m1; return; }
        }
    }
    static void lock(LiSlot16 &x) {
        for (;;) {
            uint64_t m1 = x.m.load(std::memory_order_relaxed);
            if (!(m1 >> l16::SEQ_SH & 1) && x.m.compare_exchange_weak(m1, m1 + l16::SEQ_ONE, std::memory_order_acquire)) break;
            _mm_pause();
        }
        std::atomic_thread_fence(std::memory_order_release);
    }
    static void unlock(LiSlot16 &x) { x.m.fetch_add(l16::SEQ_ONE, std::memory_order_release); }
    // under the slot lock: replace fp / owner / ep (keeps seq, and ref / hwin unless set_ref)
    static void set_fields(LiSlot16 &x, uint64_t fields, int set_ref) {
        uint64_t m1 = x.m.load(std::memory_order_relaxed), n;
        do {
            n = (m1 & ~l16::FIELDS) | fields;
            if (set_ref > 0) n |= l16::REF;
            if (set_ref == 0) n &= ~l16::REF;
        } while (!x.m.compare_exchange_weak(m1, n, std::memory_order_relaxed));
    }
    bool find(uint64_t tag, uint64_t key, uint64_t *v, uint32_t *ep, LiSlot16 **slot) const {
        uint64_t i = home(key);
        for (int n = 0; n < max_probe; n++, i = (i + 1) & mask) {
            uint64_t m, val;
            read(s[i], &m, &val);
            if (l16::fp(m) == l16::FP_EMPTY) return false;
            if ((m & l16::TAG_MASK) == tag) { *v = val; if (ep) *ep = l16::ep(m); if (slot) *slot = &s[i]; return true; }
        }
        return false;
    }
    bool put(uint64_t tag, uint64_t key, uint64_t v, uint32_t ep, bool touch, uint64_t *at = nullptr) {
        const uint64_t fields = tag | (uint64_t)(ep & l16::EP_MASK) << l16::EP_SH;
        uint64_t i = home(key), free_i = ~0ULL;
        for (int n = 0; n < max_probe; n++, i = (i + 1) & mask) {
            uint64_t m, val;
            read(s[i], &m, &val);
            if ((m & l16::TAG_MASK) == tag && l16::fp(m) > l16::FP_DEL) {
                lock(s[i]);
                if ((s[i].m.load(std::memory_order_relaxed) & l16::TAG_MASK) == tag) {
                    s[i].v.store(v, std::memory_order_relaxed);
                    set_fields(s[i], fields, touch ? 1 : -1);
                    unlock(s[i]);
                    if (at) *at = i;
                    return true;
                }
                unlock(s[i]);
                return put(tag, key, v, ep, touch, at);
            }
            if (l16::fp(m) <= l16::FP_DEL && free_i == ~0ULL) free_i = i;
            if (l16::fp(m) == l16::FP_EMPTY) break;
        }
        if (free_i == ~0ULL) return false;
        LiSlot16 &x = s[free_i];
        lock(x);
        if (l16::fp(x.m.load(std::memory_order_relaxed)) <= l16::FP_DEL) {
            x.v.store(v, std::memory_order_relaxed);
            set_fields(x, fields, 1);
            unlock(x);
            count.fetch_add(1, std::memory_order_relaxed);
            if (at) *at = free_i;
            return true;
        }
        unlock(x);
        return put(tag, key, v, ep, touch, at);
    }
    // slot-level: under the slot lock, if the slot still holds tag with value expect_v (any value if !check_v), apply f
    // (0 = delete the entry, 1 = mark deleted by the log (v = 0), 2 = mark failed (v = 1 << 63)). True if applied.
    bool apply_at(uint64_t i, uint64_t tag, bool check_v, uint64_t expect_v, int f) {
        LiSlot16 &x = s[i];
        lock(x);
        uint64_t m = x.m.load(std::memory_order_relaxed);
        bool ok = (m & l16::TAG_MASK) == tag && l16::fp(m) > l16::FP_DEL &&
                  (!check_v || x.v.load(std::memory_order_relaxed) == expect_v);
        if (ok) {
            if (f == 0) { x.v.store(0, std::memory_order_relaxed); set_fields(x, l16::FP_DEL << l16::FP_SH, 0); }
            else x.v.store(f == 1 ? 0 : 1ULL << 63, std::memory_order_relaxed);
        }
        unlock(x);
        if (ok && f == 0) count.fetch_sub(1, std::memory_order_relaxed);
        return ok;
    }
    // by key (fingerprint + owner): f as in apply_at, applied to EVERY matching entry on the key's probe chain (find
    // returns the first; a second one with the same tag can exist only if an aliasing key's install hit max_probe before
    // reaching the first, and must not survive a log record for the key). f == 1 (log delete) skips markers.
    bool apply(uint64_t tag, uint64_t key, bool check_v, uint64_t expect_v, int f) {
        uint64_t i = home(key);
        bool any = false;
        for (int n = 0; n < max_probe; n++, i = (i + 1) & mask) {
            uint64_t m, val;
            read(s[i], &m, &val);
            if (l16::fp(m) == l16::FP_EMPTY) break;
            if ((m & l16::TAG_MASK) != tag || l16::fp(m) <= l16::FP_DEL) continue;
            if (f == 1 && !(uint32_t)val) continue;   // already a marker
            any |= apply_at(i, tag, check_v, expect_v, f);
        }
        return any;
    }
};
// handle on the slot of a found entry (either layout): the fast path's ref bit and heat window
struct LiRef {
    LiSlot *a = nullptr;
    LiSlot16 *b = nullptr;
    bool ref() const { return a ? a->ref.load(std::memory_order_relaxed) : (b->m.load(std::memory_order_relaxed) & l16::REF); }
    void set_ref() const {
        if (a) a->ref.store(1, std::memory_order_relaxed);
        else b->m.fetch_or(l16::REF, std::memory_order_relaxed);
    }
    // heat reports: true if the window w was already recorded; else records it (benign races)
    bool seen_window(uint16_t w) const {
        if (a) {
            if (a->hwin.load(std::memory_order_relaxed) == w) return true;
            a->hwin.store(w, std::memory_order_relaxed);
            return false;
        }
        uint64_t m1 = b->m.load(std::memory_order_relaxed);
        if ((m1 & l16::HWIN_MASK) == (w & l16::HWIN_MASK)) return true;
        b->m.compare_exchange_strong(m1, (m1 & ~l16::HWIN_MASK) | (w & l16::HWIN_MASK), std::memory_order_relaxed);
        return false;
    }
};
// The host's local index: the 32 B layout (key stored) or, with Config::li16, the 16 B one. Same calls; owner = the
// key's owner (only the 16 B layout uses it).
struct LiIndex {
    bool c16 = false;
    uint64_t nfp = l16::FP_MASK - 1;   // fingerprint values (Config::li16_fp_bits; tests shrink it to force aliasing)
    uint32_t ep_mask = l16::EP_MASK;   // ep bits kept (Config::li16_ep_bits; tests shrink it to exercise the wrap clear)
    LiTable a;
    LiTable16 b;
    // Config::log_bloom_bits: Bloom filter over the items installed since the last rebuild --
    // the key (32 B layout) or tag(key, owner) (li16), i.e. what the table stores. put() adds the item before the table
    // install (one step w.r.t. the poller: under the log scheme every put holds poll_mu); erase / mark_deleted / sweep
    // never remove it; bloom_rebuild() only between two poll rounds, holding poll_mu.
    std::unique_ptr<std::atomic<uint64_t>[]> bloom;
    uint64_t bloom_mask = 0;   // bits - 1
    uint64_t bloom_adds = 0;   // items that set new bits since the last rebuild (poll_mu)
    void init(size_t slots, bool compact, int fp_bits, int ep_bits, uint32_t bloom_bits = 0) {
        c16 = compact;
        nfp = std::min<uint64_t>(1ULL << fp_bits, l16::FP_MASK - 1);
        ep_mask = (uint32_t)((1ULL << ep_bits) - 1) & (uint32_t)l16::EP_MASK;
        if (c16) b.init(slots); else a.init(slots);
        if (bloom_bits) {
            uint64_t bits = 64;
            while (bits < (uint64_t)slots * bloom_bits) bits <<= 1;
            bloom.reset(new std::atomic<uint64_t>[bits / 64]);
            for (uint64_t i = 0; i < bits / 64; i++) bloom[i].store(0, std::memory_order_relaxed);
            bloom_mask = bits - 1;
        }
    }
    uint64_t tag(uint64_t key, int o) const { return l16::tag(key, o, nfp); }
    uint64_t bloom_item(uint64_t key, int o) const { return c16 ? tag(key, o) : key; }
    bool bloom_has(uint64_t x) const {
        uint64_t h = mix64(x ^ 0x51afd7ed558ccd1dULL);
        for (int k = 0; k < 3; k++, h = (h >> 21) | (h << 43))
            if (!(bloom[(h & bloom_mask) >> 6].load(std::memory_order_relaxed) & (1ULL << (h & 63)))) return false;
        return true;
    }
    void bloom_set(uint64_t x) {
        uint64_t h = mix64(x ^ 0x51afd7ed558ccd1dULL);
        for (int k = 0; k < 3; k++, h = (h >> 21) | (h << 43))
            bloom[(h & bloom_mask) >> 6].fetch_or(1ULL << (h & 63), std::memory_order_relaxed);
    }
    // false: the key was not installed since the last rebuild, so this host holds no entry for it
    bool bloom_maybe(uint64_t key, int o) const { return !bloom || bloom_has(bloom_item(key, o)); }
    void bloom_rebuild() {   // caller holds poll_mu, between two poll rounds
        for (uint64_t i = 0; i <= bloom_mask / 64; i++) bloom[i].store(0, std::memory_order_relaxed);
        bloom_adds = 0;
        if (!c16) {
            for (uint64_t i = 0; i <= a.mask; i++) {
                uint64_t k = a.s[i].key.load(std::memory_order_relaxed);
                if (k != LI_EMPTY && k != LI_DEL) bloom_set(k);
            }
        } else {
            for (uint64_t i = 0; i <= b.mask; i++) {
                uint64_t m = b.s[i].m.load(std::memory_order_relaxed), f = l16::fp(m);
                if (f != l16::FP_EMPTY && f != l16::FP_DEL) bloom_set(m & l16::TAG_MASK);
            }
        }
    }
    int64_t count() const { return c16 ? b.count.load(std::memory_order_relaxed) : a.count.load(std::memory_order_relaxed); }
    uint32_t ep_of(uint32_t e) const { return c16 ? (e & ep_mask) : e; }   // what an entry's ep is compared with
    bool find(uint64_t key, int o, uint64_t *v, uint32_t *ep = nullptr, LiRef *r = nullptr) const {
        if (!c16) { LiSlot *sl = nullptr; bool f = a.find(key, v, ep, r ? &sl : nullptr); if (f && r) *r = {sl, nullptr}; return f; }
        LiSlot16 *sl = nullptr;
        bool f = b.find(tag(key, o), key, v, ep, r ? &sl : nullptr);
        if (f && r) *r = {nullptr, sl};
        return f;
    }
    bool put(uint64_t key, int o, uint64_t v, uint32_t ep = 0, bool touch = false, uint64_t *at = nullptr) {
        if (bloom) {   // before the table install (Config::log_bloom_bits)
            uint64_t x = bloom_item(key, o);
            if (!bloom_has(x)) { bloom_set(x); bloom_adds++; }
        }
        return c16 ? b.put(tag(key, o), key, v, ep & ep_mask, touch, at) : a.put(key, v, ep, touch);
    }
    bool erase(uint64_t key, int o, uint64_t expect_v = 0) {
        return c16 ? b.apply(tag(key, o), key, expect_v != 0, expect_v, 0) : a.erase(key, expect_v);
    }
    bool mark_deleted(uint64_t key, int o) { return c16 ? b.apply(tag(key, o), key, false, 0, 1) : a.mark_deleted(key); }
    bool mark_failed(uint64_t key, int o, uint64_t expect_v) {
        return c16 ? b.apply(tag(key, o), key, true, expect_v, 2) : a.mark_failed(key, expect_v);
    }
};
// the executor thread's last PUT: when it removed / replaced the old AI entry (lease) and the TSC right after its cut
// (log: unpublish + append). Per thread: with put_serial off (M5 ablation) two executors run PUTs of one owner.
static thread_local uint64_t tl_cut_ns = 0, tl_cut_tsc = 0;
static thread_local uint64_t tl_cut_mono = 0;   // the log cut on CLOCK_MONOTONIC (Config::on_cut)
static thread_local bool tl_xbump_off = false;  // ablation xtra_kill_on_put = false: no directory bump for tl_xbump_key
static thread_local uint64_t tl_xbump_key = 0;   // ... inside this PUT (other keys evicted meanwhile still bump)
static thread_local int tl_lane = 0;   // the executor thread's lane (Host::lanes index)
constexpr uint64_t AI_NONE = ~0ULL;   // Obj::ai_idx of an object whose AI entry was removed (PUT in progress)
struct Obj {
    uint64_t key, off, ai_idx;
    uint32_t gen, ver;
    int cls;
    uint32_t nl = 0;       // lines of the object
    uint64_t gc = 0;       // allocation count of its generation (true count; gen = gc mod 2^gen_bits)
    uint64_t charge = 0, obj_bytes = 0, li_bytes = 0;
    bool pinned = false;   // PUT replacing this object: skipped by eviction
    Obj *prev = nullptr, *next = nullptr;   // SIEVE list: head = newest; prev points toward the head
    Obj *aprev = nullptr, *anext = nullptr; // age list (R2 restamp): head = oldest generation
};
struct Mailbox {
    std::atomic<int> ready{0};
    std::atomic<uint32_t> seq{0};   // current request; replies carrying another seq (late, after a timeout) are dropped
    uint32_t val = 0, val2 = 0;
    char *buf = nullptr;   // SHARD: where the owner's reply value is copied (val = vlen, val2 = 1 hit / 0 miss)
    uint64_t stamp = 0;    // diag_put_timing: TSC when the owner sent the PUT's reply
};
// diag_put_timing: TSC stamps of one PUT (client send, owner receive, executor start, unpublish, cut, backend, publish)
struct PutStamps { uint64_t send = 0, recv = 0, start = 0, unpub = 0, cut = 0, backend = 0, publish = 0; };
struct PutJob { uint64_t key; int src, reqid; uint32_t seq; uint64_t t_send = 0, t_recv = 0; int op = W_PUT; };   // op: WriteOp
struct PutDone {   // lease / log: PUT returns at `due` (log: and not before the TSC reaches due_tsc)
    std::chrono::steady_clock::time_point due;
    PutJob j;
    uint32_t ver;
    uint64_t due_tsc = 0;
    PutStamps ps{};
};
static thread_local PutStamps tl_ps;               // the executor's current PUT (do_put fills unpub / cut / backend)
static thread_local uint64_t tl_reply_stamp = 0;   // put_finish -> deliver (local) / M_PUTDONE -> deliver (remote)
static inline void diag_put_add(Stats &S, int k, uint64_t a, uint64_t b);
static void diag_put_record(Stats &S, const PutStamps &p, uint64_t due, uint64_t reply) {   // owner-side segments 0..7
    if (p.send && p.recv) diag_put_add(S, 0, p.send, p.recv);
    if (p.recv && p.start) diag_put_add(S, 1, p.recv, p.start);
    if (p.start && p.unpub) diag_put_add(S, 2, p.start, p.unpub);
    if (p.cut && p.backend && p.cut >= p.backend) {   // swap order (put_cut_at_publish): the cut follows
        // the backend write; keep the segments' meaning: 3 = the swap's own work up to the cut (kg, switch, records),
        // 4 = the backend write (from the pending mark), 5 = after the cut up to the publish stamp
        if (p.unpub) diag_put_add(S, 4, p.unpub, p.backend);
        diag_put_add(S, 3, p.backend, p.cut);
        if (p.publish) diag_put_add(S, 5, p.cut, p.publish);
    } else {
        if (p.unpub && p.cut) diag_put_add(S, 3, p.unpub, p.cut);
        if (p.cut && p.backend) diag_put_add(S, 4, p.cut, p.backend);
        if (p.backend && p.publish) diag_put_add(S, 5, p.backend, p.publish);
    }
    if (p.publish) diag_put_add(S, 6, p.publish, reply);
    if (due) diag_put_add(S, 7, due, reply);
}
static inline void diag_put_add(Stats &S, int k, uint64_t a, uint64_t b) {
    uint64_t c = b > a ? b - a : 0;
    S.diag_put_hist[k][c ? std::min(47, 64 - __builtin_clzll(c)) : 0].fetch_add(1, std::memory_order_relaxed);
    S.diag_put_sum[k].fetch_add(c, std::memory_order_relaxed);
}
// log, reader side: this host's view of one owner's log. Written by the poller under Host::poll_mu; applied and drop_gen
// are also read without the lock (s0 snapshot, hit check).
struct LogView {
    bool seen = false;
    uint32_t epoch = 0;
    uint64_t pos = 0;          // records of the owner's current epoch consumed
    uint64_t count = 0;        // owner's allocation counter at the last poll
    uint64_t exam_lb = 0;      // every live entry of this owner was examined (or installed) at counter >= exam_lb
    uint64_t pass_start = 0;   // counter when the current sweep pass started
    uint64_t objs = 0;         // objects the owner caches (from its tail line; local-index cap)
    std::atomic<uint64_t> applied{0};   // records applied (monotone across epochs): s0
    std::atomic<uint32_t> drop_gen{1};  // drop-all generation; a local entry is valid only while its ep equals this
    std::vector<uint64_t> rec_seq;      // per key bucket: `applied` value of the last record applied for it
};
// PRIV / SHARD: one host's private cache in its own DRAM. Same structures as the shared area (a slab of
// [gen | payload] lines, an index of {off, gen, class}), so reads are lock-free with the same one-load-per-line
// check; writers (fill, eviction, invalidation) hold mu.
struct PObj {
    uint64_t key, off, charge;
    uint32_t gen;
    int cls;
    PObj *prev = nullptr, *next = nullptr;
};
struct Priv {
    std::mutex mu;
    std::unordered_map<uint64_t, PObj *> objs;
    std::unordered_map<uint64_t, uint32_t> kg;
    std::unique_ptr<std::atomic<uint32_t>[]> kgb;   // shard_direct: kg per bucket (kg_buckets words), lock-free reads
    std::unordered_set<uint64_t> pending;
    PObj *head = nullptr, *tail = nullptr, *hand = nullptr;
    uint64_t used = 0, bump = 0;
    uint32_t genc = 0;
    char *slab = nullptr;
    size_t slab_bytes = 0;
    std::vector<std::vector<uint64_t>> freel;
    std::unique_ptr<std::atomic<uint8_t>[]> visited;   // per slab line
    LiTable index;
};
// Monotonic reads (Config::mono): one bucket of a host's table of recent backend / slow-path returns, key -> kg need
// (floor + 1); one 64 B line, a seqlock (writers: this host's client threads; readers: its GETs, lock-free).
// key[i] = key + 1 (0 = empty slot); floor[i] = the need; times in mono_now() units (TSC >> 12).
struct alignas(64) MonoBucket {
    std::atomic<uint32_t> seq{0};
    std::atomic<uint32_t> ovf{0};       // overflowed until this time (a record found the 3 entries live): 0 = no
    std::atomic<uint64_t> key[3] = {};
    std::atomic<uint32_t> floor[3] = {}, until[3] = {};
};
static_assert(sizeof(MonoBucket) == 64, "MonoBucket");
static thread_local uint64_t tl_be_done_tsc = 0;   // Config::mono: when this executor's last backend write completed (FD check)
struct Host {
    int id = 0;
#if RC_MONO
    std::unique_ptr<MonoBucket[]> mono;   // monotonic-read table (Config::mono), Config::mono_buckets buckets
#endif
    Priv priv;
    Stats own_st;
    Stats *st = &own_st;
    std::unique_ptr<Stripe[]> stripes;
    LiIndex li;
    // owner metadata (protected by mu)
    std::mutex mu;
    std::unordered_map<uint64_t, Obj *> objs;
    std::unordered_map<uint64_t, uint32_t> kg;
    std::unordered_set<uint64_t> pending;
    Obj *head = nullptr, *tail = nullptr, *hand = nullptr;
    std::vector<std::vector<uint64_t>> freel;
    uint64_t bump = 0, used = 0;        // used = obj_bytes + li_bytes + ts_bytes (AI table is subtracted up front)
    uint64_t obj_bytes = 0, li_bytes = 0, ts_bytes = 0;
    std::deque<uint64_t> ts_expiry;     // one entry per PUT: when its tombstones stop being charged
    int sweep_pos = 0;
    uint64_t next_sweep = 0;
    std::vector<uint8_t> ai_used;                         // host-local mirror of AI slot states
    // emu_noncoherent: this host's software line cache, key = owner << 40 | slab offset
    std::mutex emu_mu;
    std::unordered_map<uint64_t, std::array<char, 64>> emu;
    std::unique_ptr<std::atomic<uint32_t>[]> slot_gen;   // per slab line: gen of the object starting there
    std::unique_ptr<std::atomic<uint8_t>[]> visited;     // per slab line: SIEVE visited bit
    // messaging (sender side, per receiver)
    std::vector<std::unique_ptr<std::mutex>> out_mu;
    std::vector<std::deque<std::vector<uint8_t>>> overflow;
    std::vector<uint64_t> head_local, tail_cache;
    std::atomic<int> overflow_n{0};
    // receiver side, per sender
    std::vector<uint64_t> rx_tail;
    // mailboxes of this host's clients
    std::mutex mb_mu;
    std::vector<std::unique_ptr<Mailbox>> mbox;
    // PUT executor
    // PUT executor lanes: exec_lanes executor threads per owner, a key's lane chosen by its kg
    // bucket, so PUTs of one key (and of keys sharing a kg word) stay serial (M5; the kg word keeps one writer). Every
    // scheme (log, lease, invalidate, priv, shard); one lane with put_serial off (the M5 ablation's two threads share it).
    struct Lane {
        std::mutex ex_mu;
        std::condition_variable ex_cv;
        std::deque<PutJob> jobs;
        std::atomic<uint32_t> njobs{0};   // jobs.size(), read without ex_mu (put_async: the asynchronous executor's spin)
        std::deque<PutDone> done_q;    // completions waiting for T / W, in due order
        std::thread th;
        // the lane's current PUT that waits for acks (invalidate: revoke; priv: round 1); its id carries the lane in
        // the low 8 bits, so an M_ACK reaches the lane that sent the request (per-lane state)
        uint64_t putseq = 0;
        std::atomic<uint64_t> cur_put{0};
        std::atomic<int> acks_needed{0};
        std::atomic<uint32_t> acked{0};      // revoke_timeout_ns: hosts that acked the current PUT's revoke
    };
    std::vector<std::unique_ptr<Lane>> lanes;
    std::atomic<int> executing{0};
    std::atomic<uint32_t> ack_dead{0};   // hosts declared dead by revoke_timeout_ns (never waited for again)
    std::thread handler, executor2, ticker;   // executor2: only with put_serial off (M5 ablation; lane 0)
    // lease
    std::atomic<uint32_t> epoch{1};
    std::atomic<uint64_t> last_tick_tsc{0};   // TSC of the scheduled time of the current epoch (base + e * period)
    uint64_t base_tsc = 0;                     // epoch e covers [base + e * period, base + (e+1) * period)
    std::atomic<int> epoch_paused{0}, unreachable{0};
    std::atomic<uint64_t> writable_at{0};   // mono_ns: restart_wait
    // delayed reuse (Config::reuse_delayed), owner side (under mu): slots waiting for the other hosts' acks
    struct Reclaim { uint64_t key, off; int cls; uint32_t need; uint64_t obj_bytes, li_bytes, t; };
    std::unordered_map<uint64_t, Reclaim> reclaims;
    uint64_t reclaim_seq = 0, next_gauge = 0, pending_bytes = 0;
    uint32_t dead_mask = 0;
    // log, owner side (under mu)
    uint64_t gcount = 0;                    // allocation counter (true count)
    uint64_t log_tail = 0;                  // records appended in this epoch
    uint32_t log_epoch = 0;
    Obj *age_head = nullptr, *age_tail = nullptr;
    // log, reader side
    std::unique_ptr<LogView[]> lv;          // per owner
    std::vector<TailEnt> tail_buf;          // log_batch_tails: the tail line items of a batched read (2 per owner, poll_mu held)
    std::mutex poll_mu;                     // a poll round, and a slow path's check + install, are atomic w.r.t. each other
    std::atomic<uint64_t> hb{0};            // heartbeat: TSC just before the last completed round read the tails
    std::atomic<int> poll_paused{0};
    std::atomic<uint32_t> poll_seq{0};      // completed poll rounds (heat window)
    uint64_t li_sweep_pos = 0;              // next local-index slot of the incremental R2 sweep
    std::deque<uint64_t> clock_q;           // local-index cap (poll_mu): keys in install order; the CLOCK hand pops from the front
    std::deque<uint32_t> clock_qi;          // ... li16: slot indices in install order instead of keys (4 B per held entry)
    // Config::li_clock_inq (under poll_mu): li16: 1 = the slot index is in clock_qi; 32 B entries: the keys in clock_q
    std::vector<uint8_t> clock_inq_i;
    std::unordered_set<uint64_t> clock_inq_k;
    std::vector<uint32_t> kgb;              // kg_cxl, owner side (under mu): its kg words (mirror of the table in CXL)
    std::thread poller;
};

struct Cache::Impl {
    Config cfg;
    Backend *be;
    int H;
    uint64_t budget_eff = 0;   // budget_bytes minus the AI table
    char *shm = nullptr;
    size_t shm_bytes = 0, per_owner = 0, off_ai = 0, off_rings = 0, off_tails = 0, off_gstate = 0, off_log = 0, off_kg = 0, off_xdir = 0, xdir_mask = 0,
           ring_stride = 0;
    std::vector<std::unique_ptr<Host>> hosts;
    std::atomic<bool> stop{false};

    // ---- addressing
    char *owner_base(int o) const { return shm + o * per_owner; }
    char *slab(int o) const { return owner_base(o); }
    AiEnt *ai(int o) const { return (AiEnt *)(owner_base(o) + off_ai); }
    char *ring_hdr(int s, int r) const { return owner_base(s) + off_rings + r * ring_stride; }
    char *ring_line(int s, int r, uint64_t i) const { return ring_hdr(s, r) + LINE + (i & (cfg.ring_lines - 1)) * LINE; }
    char *tail_line(int r, int s) const { return owner_base(r) + off_tails + s * LINE; }   // in receiver r's area
    Stripe &stripe(int h, uint64_t k) { return hosts[h]->stripes[mix64(k) % NSTRIPE]; }
    uint64_t ai_home(uint64_t k) const { return mix64(k ^ 0x5bd1e995ULL) & (cfg.ai_slots - 1); }

    struct Layout { size_t slab, off_ai, off_rings, off_tails, off_gstate, off_log, off_kg, off_xdir, xdir_slots, ring_stride, per_owner, total; };
    static Layout layout(const Config &c) {
        Layout L;
        L.slab = c.slab_bytes & ~(uint64_t)(LINE - 1);
        L.ring_stride = LINE + c.ring_lines * LINE;
        L.off_ai = L.slab;
        L.off_rings = (L.off_ai + c.ai_slots * sizeof(AiEnt) + LINE - 1) & ~(size_t)(LINE - 1);
        L.off_tails = L.off_rings + c.hosts * L.ring_stride;
        L.off_gstate = L.off_tails + c.hosts * LINE;   // GSTATE baselines: one state word per owner
        L.off_log = L.off_gstate + LINE;               // log: tail line, then log_lines record lines
        L.off_kg = L.off_log + LINE + c.log_lines * LINE;   // kg table (kg_cxl), one u32 word per bucket
        L.off_xdir = (L.off_kg + c.kg_buckets * 4 + LINE - 1) & ~(size_t)(LINE - 1);   // XTRA directory
        L.xdir_slots = 0;
        if (c.scheme == Scheme::XTRA) {
            L.xdir_slots = 1;
            while (L.xdir_slots * 2 * c.hosts <= c.xtra_dir_slots) L.xdir_slots *= 2;
        }
        L.per_owner = (L.off_xdir + L.xdir_slots * 8 * (c.xtra_dir_replicas ? c.hosts : 1) + 4095) & ~(size_t)4095;   // XREP: a copy per host
        L.total = L.per_owner * c.hosts;
        return L;
    }
    static void check_cfg(const Config &c) {
        if (c.ai_slots & (c.ai_slots - 1) || c.ring_lines & (c.ring_lines - 1)) {
            fprintf(stderr, "ai_slots and ring_lines must be powers of two\n");
            exit(2);
        }
        if (c.slab_bytes / LINE > OFF_MASK) { fprintf(stderr, "slab too large for 28-bit line offsets\n"); exit(2); }
        if (c.shard_direct && (c.scheme != Scheme::SHARD || !c.shm_name.empty() || c.local_host >= 0)) {
            fprintf(stderr, "shard_direct: only with scheme shard in threads mode (all hosts in one process)\n");
            exit(2);
        }
        if (c.li16_fp_bits < 1 || c.li16_fp_bits > 20 || c.li16_ep_bits < 1 || c.li16_ep_bits > 16) {
            fprintf(stderr, "li16_fp_bits: 1..20, li16_ep_bits: 1..16\n");
            exit(2);
        }
        if (!c.rules.put_serial && (!c.rules.log || c.rules.lease || c.rules.revoke || c.scheme == Scheme::PRIV || c.scheme == Scheme::SHARD)) {
            fprintf(stderr, "put_serial off (M5 ablation): log scheme only\n");
            exit(2);
        }
        if (c.owners < 0 || c.owners > c.hosts || c.hosts > 32) { fprintf(stderr, "need 0 <= owners <= hosts <= 32\n"); exit(2); }
        if (c.xtra_dir_replicas > 2) { fprintf(stderr, "xtra_dir_replicas: need 0, 1 or 2\n"); exit(2); }
        if (c.xtra_dir_replicas && (c.scheme != Scheme::XTRA || !c.xtra_own_freshness || !c.rules.log || c.rules.lease || c.rules.revoke)) {
            // as xtra_own(): with lease / revoke the borrowed path runs, which writes only copy 0
            fprintf(stderr, "xtra_dir_replicas (XREP): XTRA-style own freshness only (scheme xtra, xtra_own_freshness, the log, no lease / revoke)\n");
            exit(2);
        }
        if (c.xtra_dir_replicas && layout(c).xdir_slots < 8) {
            fprintf(stderr, "xtra_dir_replicas: need xdir_slots >= 8 per owner (separate cache lines for host copies)\n");
            exit(2);
        }
#if !RC_MONO
        if (c.mono) { fprintf(stderr, "mono: this build has RC_MONO=0; build with RC_MONO=1 bash proto/build.sh\n"); exit(2); }
#endif
        if (c.mono && (!c.rules.log || c.rules.lease || c.rules.revoke || !c.rules.kg_cxl || c.scheme == Scheme::PRIV ||
                       c.scheme == Scheme::SHARD || c.scheme == Scheme::VER || c.scheme == Scheme::VER_NOFLUSH ||
                       (c.scheme == Scheme::XTRA && c.xtra_own_freshness) || c.mono_buckets < 1 || (c.mono_buckets & (c.mono_buckets - 1)))) {
            fprintf(stderr, "mono: log scheme with kg in CXL, a local index with object generations (not PRIV / SHARD / VER / "
                            "XTRA own freshness); mono_buckets a power of two\n");
            exit(2);
        }
        if (c.kg_buckets < 16 || (c.kg_buckets & (c.kg_buckets - 1))) { fprintf(stderr, "kg_buckets: power of two >= 16\n"); exit(2); }
        if (c.log_lines < 1 || c.gen_bits < 4 || c.gen_bits > 32) { fprintf(stderr, "need log_lines >= 1, 4 <= gen_bits <= 32\n"); exit(2); }
        if (c.rules.log && c.log_max_alloc_per_s > 0) {   // R2 sweep margin (rackcache.h log_sweep_slots)
            const double m = sweep_margin_of(c);
            if (m < 1) {
                fprintf(stderr, "R2 sweep too slow: during one pass over the local index (%.3g s) an owner may advance "
                        "%.3g allocations + the sweep threshold 2^%d >= 2^%d (margin %.3g < 1): raise log_sweep_slots, "
                        "gen_bits or lower li_slots / log_max_alloc_per_s\n", sweep_pass_s(c),
                        c.log_max_alloc_per_s * sweep_pass_s(c), c.gen_bits - 1, c.gen_bits, m);
                exit(2);
            }
            if (m < 10) fprintf(stderr, "warning: R2 sweep margin only %.3g (pass %.3g s)\n", m, sweep_pass_s(c));
        }
        if (!c.charge_value_bytes_only && c.charge_model != CHARGE_K11B && c.scheme != Scheme::PRIV && c.scheme != Scheme::SHARD &&
            c.budget_bytes <= c.ai_slots * sizeof(AiEnt)) {   // (CHARGE_K11B: the AI entries are charged per object)
            fprintf(stderr, "budget smaller than the AI table\n");
            exit(2);
        }
    }
    static void bind_and_touch(char *p, size_t bytes, int node) {
        if (node >= 0) {
            unsigned long mask = 1UL << node;
            if (mbind(p, bytes, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT)) { perror("mbind"); exit(1); }
        }
        madvise(p, bytes, MADV_HUGEPAGE);
        memset(p, 0, bytes);
    }

    Impl(const Config &c, Backend *b) : cfg(c), be(b), H(c.hosts) {
#if !defined(__AVX512F__)
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "# WARNING: built without AVX-512F: 64 B line loads are two 32 B loads (H5 not held); "
                            "functional tests only, not for measurement\n");
        }
#endif
        if ((cfg.obj_read_mode & 2) && !mstore_available()) {   // no AVX-512 BW: fall back to the prefetch-only read
            fprintf(stderr, "# obj_read: no AVX-512 BW, masked-store copies off (%s -> %s)\n", obj_read_mode_name(cfg.obj_read_mode),
                    obj_read_mode_name(cfg.obj_read_mode & 1 ? 1 : 0));
            cfg.obj_read_mode &= 1;
        }
        check_cfg(cfg);
        budget_eff = cfg.charge_value_bytes_only || cfg.charge_model == CHARGE_K11B ? cfg.budget_bytes
                                                                                    : cfg.budget_bytes - cfg.ai_slots * sizeof(AiEnt);
        Layout L = layout(cfg);
        cfg.slab_bytes = L.slab;
        ring_stride = L.ring_stride;
        off_ai = L.off_ai;
        off_rings = L.off_rings;
        off_tails = L.off_tails;
        off_gstate = L.off_gstate;
        off_log = L.off_log;
        off_kg = L.off_kg;
        off_xdir = L.off_xdir;
        xdir_mask = L.xdir_slots ? L.xdir_slots - 1 : 0;
        per_owner = L.per_owner;
        shm_bytes = L.total;
        if (cfg.shm_name.empty()) {
            shm = (char *)mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (shm == MAP_FAILED) { perror("mmap"); exit(1); }
            bind_and_touch(shm, shm_bytes, cfg.shm_node);
        } else {
            int fd = shm_open(cfg.shm_name.c_str(), O_RDWR, 0);
            if (fd < 0) { perror("shm_open"); exit(1); }
            struct stat sb;
            if (fstat(fd, &sb) || (size_t)sb.st_size != shm_bytes) {
                fprintf(stderr, "shm %s: size %ld, expected %zu (config mismatch?)\n", cfg.shm_name.c_str(), (long)sb.st_size, shm_bytes);
                exit(1);
            }
            shm = (char *)mmap(nullptr, shm_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            if (shm == MAP_FAILED) { perror("mmap shm"); exit(1); }
            if (cfg.ro_peers) {   // ro_peers: every other owner's block read-only (per_owner is page-aligned)
                if (cfg.local_host < 0) { fprintf(stderr, "ro_peers: needs one process per host (local_host)\n"); exit(2); }
                for (int o = 0; o < H; o++)
                    if (o != cfg.local_host && mprotect(owner_base(o), per_owner, PROT_READ)) { perror("mprotect ro_peers"); exit(1); }
            }
        }
        if (cfg.ro_peers && cfg.shm_name.empty()) {
            fprintf(stderr, "ro_peers: needs one process per host (--procs); in threads mode all hosts share one address space\n");
            exit(2);
        }
        size_t slab_lines = cfg.slab_bytes / LINE;
        for (int h = 0; h < H; h++) {
            if (cfg.local_host >= 0 && h != cfg.local_host) { hosts.emplace_back(nullptr); continue; }
            auto x = std::make_unique<Host>();
            x->id = h;
#if RC_MONO
            if (cfg.mono) x->mono.reset(new MonoBucket[cfg.mono_buckets]);
#endif
            if (cfg.shared_stats) x->st = &cfg.shared_stats[h];
            x->stripes.reset(new Stripe[NSTRIPE]);
            if (cfg.li_slots & (cfg.li_slots - 1)) { fprintf(stderr, "li_slots must be a power of two\n"); exit(2); }
            if (cfg.scheme != Scheme::PRIV && cfg.scheme != Scheme::SHARD) x->li.init(cfg.li_slots, li16_on(), cfg.li16_fp_bits, cfg.li16_ep_bits, log_scheme() ? cfg.log_bloom_bits : 0);
            else {   // private slab in this host's DRAM (bound to node h with bind_hosts), index, visited bits
                Priv &V = x->priv;
                V.slab_bytes = cfg.slab_bytes;
                V.slab = (char *)mmap(nullptr, V.slab_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (V.slab == MAP_FAILED) { perror("mmap priv"); exit(1); }
                if (cfg.bind_hosts && numa_available() >= 0) {
                    unsigned long mask = 1UL << (h % (numa_max_node() + 1));
                    mbind(V.slab, V.slab_bytes, MPOL_BIND, &mask, sizeof(mask) * 8, 0);
                }
                madvise(V.slab, V.slab_bytes, MADV_HUGEPAGE);
                memset(V.slab, 0, V.slab_bytes);
                V.freel.resize(NCLS);
                V.visited.reset(new std::atomic<uint8_t>[V.slab_bytes / LINE]);
                for (size_t i = 0; i < V.slab_bytes / LINE; i++) V.visited[i] = 0;
                V.index.init(cfg.li_slots);
                V.kgb.reset(new std::atomic<uint32_t>[cfg.kg_buckets]);
                for (size_t b = 0; b < cfg.kg_buckets; b++) V.kgb[b] = 0;
            }
            x->freel.resize(NCLS);
            x->ai_used.assign(cfg.ai_slots, AI_EMPTY);
            x->slot_gen.reset(new std::atomic<uint32_t>[slab_lines]);
            x->visited.reset(new std::atomic<uint8_t>[slab_lines]);
            for (size_t i = 0; i < slab_lines; i++) { x->slot_gen[i] = 0; x->visited[i] = 0; }
            for (int r = 0; r < H; r++) x->out_mu.emplace_back(new std::mutex);
            x->overflow.resize(H);
            x->head_local.assign(H, 0);
            x->tail_cache.assign(H, 0);
            x->rx_tail.assign(H, 0);
            x->lv.reset(new LogView[H]);
            x->tail_buf.resize(2 * (size_t)H);
            x->kgb.assign(cfg.kg_buckets, 0);
            for (int o = 0; o < H; o++) x->lv[o].rec_seq.assign(REC_BUCKETS, 0);
            hosts.push_back(std::move(x));
        }
    }
    ~Impl() {
        for (auto &h : hosts) {
            if (!h) continue;
            for (PObj *q = h->priv.head; q;) { PObj *n = q->next; delete q; q = n; }
            if (h->priv.slab) munmap(h->priv.slab, h->priv.slab_bytes);
            Obj *o = h->head;
            while (o) { Obj *n = o->next; delete o; o = n; }
        }
        munmap(shm, shm_bytes);
    }

    // ---------------------------------------------------------------- objects
    // Owner write: per line gen first, then payload; clwb every line; sfence.
    void obj_write(char *o, uint32_t gen, uint64_t key, const char *val, uint32_t vlen) {
        uint32_t nl = obj_lines(vlen);
        char stream[HDR + MAX_VLEN + PAY];
        memcpy(stream, &key, 8);
        memcpy(stream + 8, &vlen, 4);
        memset(stream + 12, 0, 4);
        memcpy(stream + HDR, val, vlen);
        memset(stream + HDR + vlen, 0, PAY);
        for (uint32_t i = 0; i < nl; i++) {
            char *line = o + (size_t)i * LINE;
            vst32(line, gen);
            barrier();
            if (i == 0) hook(Hook::OwnerAfterGenStore, owner_of_ptr(o), key);
            memcpy(line + 4, stream + (size_t)i * PAY, PAY);
            barrier();
            wb_line(line);
        }
        _mm_sfence();
    }
    int owner_of_ptr(const char *p) const { return (int)((p - shm) / per_owner); }
    // One object line as seen by reader host h: one aligned 64 B load (or, in emu mode, the host's cached copy).
    void fetch_line(int h, int o, uint64_t off, char *dst) {
        const char *src = slab(o) + off;
        tl_shm.rd++;
        if (!cfg.emu_noncoherent) { load64(src, dst); return; }
        Host &X = *hosts[h];
        uint64_t k = (uint64_t)o << 40 | off;
        std::lock_guard<std::mutex> g(X.emu_mu);
        auto it = X.emu.find(k);
        if (it == X.emu.end()) {
            std::array<char, 64> a;
            load64(src, a.data());
            it = X.emu.emplace(k, a).first;
        }
        memcpy(dst, it->second.data(), LINE);
    }
    // Reader: every line is ONE 64 B load; gen is taken from the loaded copy; line 0 must hold the key.
    // max_lines = upper bound from the index entry (size class). Returns false on any mismatch.
    // Config::obj_read_mode: bit 0 = prefetcht0 lines
    // 1..max_lines-1 first (objects of >= obj_prefetch_min_lines lines), so the per-line copies do not fill the
    // out-of-order window before the later lines' loads issue; bit 1 = copy each line's payload with one masked AVX-512
    // store from the 64 B load's register (not in emu mode). Either way: one 64 B load per line, generation checked on
    // every line, key on line 0; in the slow path the prefetch comes after the flush (obj_read runs after obj_flush).
    // gen_out (XTRA's Coordination Directory versions): with gen == 0 the generation is taken from line 0 (nonzero) and every
    // later line must carry the same one (torn / reused slots still fail); the generation found is returned in *gen_out
    bool obj_read(int h, int o, uint64_t off, uint32_t gen, uint64_t key, uint32_t max_lines, char *out,
                  uint32_t *vlen_out, uint32_t *gen_out = nullptr) {
        if (off % LINE || off + LINE > cfg.slab_bytes || (gen == 0 && !gen_out)) return false;
        if ((cfg.obj_read_mode & 1) && max_lines >= cfg.obj_prefetch_min_lines) {
            const char *p = slab(o) + off;
            uint64_t pl = std::min<uint64_t>(max_lines, (cfg.slab_bytes - off) / LINE);
            for (uint64_t i = 1; i < pl; i++) _mm_prefetch(p + i * LINE, _MM_HINT_T0);
        }
        if ((cfg.obj_read_mode & 2) && !cfg.emu_noncoherent) return obj_read_mstore(o, off, gen, key, max_lines, out, vlen_out, gen_out);
        alignas(64) char l[LINE];
        fetch_line(h, o, off, l);
        uint32_t g;
        uint64_t k;
        uint32_t vlen;
        memcpy(&g, l, 4);
        memcpy(&k, l + 4, 8);
        memcpy(&vlen, l + 12, 4);
        if (!gen) gen = g;   // gen_out mode
        if (g != gen || !g || k != key || vlen > MAX_VLEN) return false;
        uint32_t nl = obj_lines(vlen);
        if ((max_lines && nl > max_lines) || off + (uint64_t)nl * LINE > cfg.slab_bytes) return false;
        uint32_t done = std::min<uint32_t>(vlen, PAY - HDR);
        memcpy(out, l + 4 + HDR, done);
        for (uint32_t i = 1; i < nl; i++) {
            fetch_line(h, o, off + (uint64_t)i * LINE, l);
            memcpy(&g, l, 4);
            if (g != gen) return false;
            uint32_t n = std::min<uint32_t>(vlen - done, PAY);
            memcpy(out + done, l + 4, n);
            done += n;
        }
        *vlen_out = vlen;
        if (gen_out) *gen_out = gen;
        return true;
    }
    bool obj_read_mstore(int o, uint64_t off, uint32_t gen, uint64_t key, uint32_t max_lines, char *out, uint32_t *vlen_out,
                         uint32_t *gen_out = nullptr) {
#if !defined(__AVX512BW__)
        (void)o; (void)off; (void)gen; (void)key; (void)max_lines; (void)out; (void)vlen_out; (void)gen_out;
        return false;   // never called: the constructor drops the mstore bit without AVX-512 BW
#else
        const char *p = slab(o) + off;
        __m512i x = vload64(p);
        tl_shm.rd++;
        alignas(64) char l[LINE];
        _mm512_store_si512((void *)l, x);
        uint32_t g, vlen;
        uint64_t k;
        memcpy(&g, l, 4);
        memcpy(&k, l + 4, 8);
        memcpy(&vlen, l + 12, 4);
        if (!gen) gen = g;   // gen_out mode
        if (g != gen || !g || k != key || vlen > MAX_VLEN) return false;
        uint32_t nl = obj_lines(vlen);
        if ((max_lines && nl > max_lines) || off + (uint64_t)nl * LINE > cfg.slab_bytes) return false;
        uint32_t done = std::min<uint32_t>(vlen, PAY - HDR);
        memcpy(out, l + 4 + HDR, done);
        for (uint32_t i = 1; i < nl; i++) {
            x = vload64(p + (size_t)i * LINE);
            tl_shm.rd++;
            if ((uint32_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(x)) != gen) return false;
            uint32_t n = std::min<uint32_t>(vlen - done, PAY);
            __mmask64 m = (n >= PAY ? ~0ULL : ((1ULL << (4 + n)) - 1)) & ~0xFULL;   // bytes 4 .. 4 + n of the line
            _mm512_mask_storeu_epi8(out + done - 4, m, x);
            done += n;
        }
        *vlen_out = vlen;
        if (gen_out) *gen_out = gen;
        return true;
#endif
    }
    // reader host h flushes nl lines of an object: clflushopt each line, then mfence
    void obj_flush(int h, int o, uint64_t off, uint32_t nl) {
        if (off % LINE || off + (uint64_t)nl * LINE > cfg.slab_bytes || nl > 192) return;
        for (uint32_t i = 0; i < nl; i++) flush_line(slab(o) + off + (size_t)i * LINE);
        _mm_mfence();
        if (cfg.emu_noncoherent) {
            Host &X = *hosts[h];
            std::lock_guard<std::mutex> g(X.emu_mu);
            for (uint32_t i = 0; i < nl; i++) X.emu.erase((uint64_t)o << 40 | (off + (uint64_t)i * LINE));
        }
    }

    // ---------------------------------------------------------------- authoritative index
    struct AiView { uint64_t off; uint32_t gen, kg, nl; };
    // Reader lookup: flush + mfence the line holding each probed entry, then one 16 B load. 0 = absent, 1 = found.
    // A fingerprint match is taken as the key's entry; a collision only costs a miss (line 0 holds the full key).
    int ai_lookup(int o, uint64_t key, AiView *v, bool flush = true) {
        AiEnt *t = ai(o);
        uint32_t fp = key_fp(key);
        uint64_t idx = ai_home(key);
        for (int i = 0; i < cfg.max_probe; i++, idx = (idx + 1) & (cfg.ai_slots - 1)) {
            AiEnt *e = &t[idx];
            if (i == 0 || ((uintptr_t)e % LINE) == 0) {
                tl_shm.rd++;   // a new line of the AI table
                if (flush) {
                    flush_line(e);
                    _mm_mfence();
                }
            }
            AiEnt x = load16(e);
            if (x.gen == 0 && x.w0 == 0) return 0;      // empty: end of the probe chain
            if (x.gen == 0) continue;                    // deleted
            if (x.fp == fp) {
                uint32_t cls = x.w0 >> 28;
                if (cls >= (uint32_t)NCLS) return 0;
                *v = {(uint64_t)(x.w0 & OFF_MASK) * LINE, x.gen, x.kg, CLS[cls]};
                return 1;
            }
        }
        return 0;
    }
    // the XTRA directory slot of slab offset off in owner o's region
    uint64_t *xdir(int o, uint64_t off) const {
        return (uint64_t *)(owner_base(o) + off_xdir) + (mix64(off / LINE ^ 0x2545f4914f6cdd1dULL) & xdir_mask);
    }
    // reader: flush + load the slot before reading the object (XTRA on CXL 2.0: no coherence for the directory)
    bool xdir_check(int o, int h, uint64_t off, uint32_t gen, ClientStats &cs) {   // h: the reader (XREP: its copy)
        uint64_t *w = xrep(xdir(o, off), h);
        flush_line(w);
        _mm_mfence();
        tl_shm.rd++;
        cs.xdir_reads++;
        if (vld64(w) == gen) return true;
        cs.xdir_other++;   // borrowed: a collision or a newer write, the object's lines decide; own freshness: slow path
        return false;
    }
    // With xtra_own_freshness the directory is XTRA's Coordination Directory (CD; XTRA §4.1 / §4.3): each word is a
    // VERSION COUNTER of its slot, and a host's local entry keeps the version it saw (XTRA's shadow version) in the field
    // that holds the generation in the other schemes (li_pack's 32-bit field: the version's low 32 bits; the owner skips values
    // whose low 32 bits are 0, which mark deleted entries). The slot is chosen by KEY, not by slab offset: a PUT here installs
    // the new version at a NEW offset, so the key is the object's identity (XTRA updates in place), and the slow path can
    // read the slot BEFORE the authoritative entry (any publish after that read moves the version, so the entry fails its
    // next hit once). The owner (mu held, the only writer of its slots) moves a slot's version whenever an object of that
    // slot is published (install, restamp: ai_write), replaced by a PUT, or evicted -- each after the AI change is written
    // back. Keys sharing a slot each revalidate once per change (XTRA: clear the slot's tracked entries); under a generation
    // comparison instead, the object that lost the slot would fail every hit until it was republished.
    uint64_t *xslot(int o, uint64_t key) const {
        return (uint64_t *)(owner_base(o) + off_xdir) + (mix64(key ^ 0x9e3779b97f4a7c15ULL) & xdir_mask);
    }
    // XREP (Config::xtra_dir_replicas): host h's copy of directory word w (w itself without the copies)
    uint64_t *xrep(uint64_t *w, int h) const { return cfg.xtra_dir_replicas ? w + (size_t)h * (xdir_mask + 1) : w; }
    void xver_bump(int o, uint64_t key) {   // owner, mu held; the caller fences (sfence / mfence) before replying
        if (tl_xbump_off && key == tl_xbump_key) return;
        uint64_t *w = xslot(o, key);
        uint64_t v = __atomic_load_n(w, __ATOMIC_RELAXED);
        do v++; while ((uint32_t)v == 0);
        if (cfg.xtra_dir_replicas == 2) {
            // XREP2 (xtra_dir_replicas = 2): evict every copy, including the counter read by the owner's next bump.
            // The caller's existing fence completes these flushes before publishing the reply.
            for (int h = 0; h < cfg.hosts; h++) {
                uint64_t *r = xrep(w, h);
                __atomic_store_n(r, v, __ATOMIC_RELAXED);
                flush_line(r);
            }
            return;
        }
        __atomic_store_n(w, v, __ATOMIC_RELAXED);
        wb_line(w);
        if (cfg.xtra_dir_replicas)   // XREP: every host's copy, where the single word was written (before the caller's fence)
            for (int h = 1; h < cfg.hosts; h++) {
                uint64_t *r = xrep(w, h);
                __atomic_store_n(r, v, __ATOMIC_RELAXED);
                wb_line(r);
            }
    }
    uint32_t xver_load(int o, int h, uint64_t key, ClientStats &cs) {   // reader h: flush + mfence + load the key's slot
        uint64_t *w = xrep(xslot(o, key), h);
        flush_line(w);
        _mm_mfence();
        tl_shm.rd++;
        cs.xdir_reads++;
        return (uint32_t)vld64(w);
    }
#if RC_MONO
    // ---- monotonic reads (Config::mono): the host's table of floors
    static inline uint32_t mono_now() { return (uint32_t)(__rdtsc() >> 12); }
    uint32_t mono_hold() const {   // FD + P in mono_now() units (rounded up); 0 with the ablation mono_hold off (expires at once)
        if (!cfg.rules.mono_hold) return 0;
        return (uint32_t)((double)(cfg.mono_fd_ns + cfg.log_poll_ns) * tsc_per_ns() / 4096.0) + 2;
    }
    MonoBucket &mono_bucket(Host &X, uint64_t key) const { return X.mono[mix64(key ^ 0x2b992ddfa23249d6ULL) & (cfg.mono_buckets - 1)]; }
    // The table stores NEED = the lowest kg a cached copy may have to be served (= floor + 1, no underflow at kg 0):
    // a backend return records kg-loaded-after-the-read + 1, a slow-path return the returned object's kg.
    // the GET's snapshot at its start: 0 = not in the table, 1 = in it (*floor = need), 2 = its bucket is overflowed (take the backend)
    // (a slot holds key + 1: 0 = empty, so key 0 is an ordinary key)
    __attribute__((noinline)) int mono_snap(Host &X, uint64_t key, uint32_t *floor) const {
        MonoBucket &b = mono_bucket(X, key);
        const uint64_t tag = key + 1;
        for (;;) {
            uint32_t s1 = b.seq.load(std::memory_order_acquire);
            if (s1 & 1) { _mm_pause(); continue; }
            int idx = -1;
            uint32_t fl = 0, un = 0, ov = b.ovf.load(std::memory_order_relaxed);
            for (int i = 0; i < 3; i++)
                if (b.key[i].load(std::memory_order_relaxed) == tag) {
                    idx = i;
                    fl = b.floor[i].load(std::memory_order_relaxed);
                    un = b.until[i].load(std::memory_order_relaxed);
                    break;
                }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (b.seq.load(std::memory_order_relaxed) != s1) continue;
            if (idx < 0 && !ov) return 0;   // the common case: no clock read
            const uint32_t now = mono_now();
            if (idx >= 0 && (int32_t)(un - now) > 0) { *floor = fl; return 1; }
            if (ov && (int32_t)(ov - now) > 0) return 2;
            return 0;
        }
    }
    // a backend or slow-path return of this host records (key, floor); never drops a live entry (a full bucket overflows)
    void mono_record(Host &X, uint64_t key, uint32_t floor, ClientStats &cs) {
        MonoBucket &b = mono_bucket(X, key);
        const uint64_t tag = key + 1;
        for (;;) {
            uint32_t s1 = b.seq.load(std::memory_order_relaxed);
            if (!(s1 & 1) && b.seq.compare_exchange_weak(s1, s1 + 1, std::memory_order_acquire)) break;
            _mm_pause();
        }
        std::atomic_thread_fence(std::memory_order_release);
        const uint32_t now = mono_now(), until = now + mono_hold();
        int idx = -1, fr = -1;
        for (int i = 0; i < 3; i++) {
            uint64_t k = b.key[i].load(std::memory_order_relaxed);
            if (k == tag) { idx = i; break; }
            if (fr < 0 && (k == 0 || (int32_t)(b.until[i].load(std::memory_order_relaxed) - now) <= 0)) fr = i;
        }
        if (idx >= 0) {
            const bool live = (int32_t)(b.until[idx].load(std::memory_order_relaxed) - now) > 0;
            const uint32_t f0 = b.floor[idx].load(std::memory_order_relaxed);
            b.floor[idx].store(live ? std::max(f0, floor) : floor, std::memory_order_relaxed);
            b.until[idx].store(until, std::memory_order_relaxed);
        } else if (fr >= 0) {
            b.key[fr].store(tag, std::memory_order_relaxed);
            b.floor[fr].store(floor, std::memory_order_relaxed);
            b.until[fr].store(until, std::memory_order_relaxed);
        } else {   // all 3 live: the bucket's other keys take the backend until it expires
            b.ovf.store(until, std::memory_order_relaxed);
            cs.mono_overflow++;
        }
        b.seq.fetch_add(1, std::memory_order_release);
        cs.mono_records++;
    }
#endif
    void ai_write(int o, uint64_t idx, uint32_t gen, uint32_t kgv, uint64_t key, uint64_t off, int cls) {
        AiEnt *e = &ai(o)[idx];
        store16(e, AiEnt{(uint32_t)(off / LINE) | (uint32_t)cls << 28, gen, kgv, key_fp(key)});
        wb_line(e);
        if (cfg.scheme == Scheme::XTRA && xtra_own()) {   // CD versions: the AI entry reaches memory before the version moves
            _mm_sfence();
            xver_bump(o, key);
        } else if (cfg.scheme == Scheme::XTRA) {   // publish the object's generation in its directory slot too
            uint64_t *w = xdir(o, off);
            __atomic_store_n(w, (uint64_t)gen, __ATOMIC_RELAXED);
            wb_line(w);
        }
        _mm_sfence();
    }
    void ai_delete(int o, uint64_t idx) {
        AiEnt *e = &ai(o)[idx];
        store16(e, AiEnt{DELETED_W0, 0, 0, 0});
        wb_line(e);
        _mm_sfence();
        hosts[o]->ai_used[idx] = AI_DELETED;
    }
    // owner: find a free AI slot within the reader probe limit
    bool ai_alloc(int o, uint64_t key, uint64_t *idx_out) {
        Host &X = *hosts[o];
        uint64_t idx = ai_home(key);
        for (int i = 0; i < cfg.max_probe; i++, idx = (idx + 1) & (cfg.ai_slots - 1))
            if (X.ai_used[idx] != AI_FULL) { *idx_out = idx; return true; }
        return false;
    }

    // ---------------------------------------------------------------- owner: generations, R2 restamp, log
    bool log_scheme() const {
        return cfg.rules.log && cfg.scheme != Scheme::PRIV && cfg.scheme != Scheme::SHARD;
    }
    // the swap write order (Rules::put_cut_at_publish) applies to the log scheme only (SHARD has its own)
    bool swap_on_put() const {
        return (cfg.rules.put_cut_at_publish || xtra_own()) && log_scheme() && !cfg.rules.lease && !cfg.rules.revoke;
    }
    // XTRA-style with its own freshness check (Config::xtra_own_freshness): no records, no heartbeat
    bool xtra_own() const {
        return cfg.xtra_own_freshness && cfg.scheme == Scheme::XTRA && cfg.rules.log && !cfg.rules.lease && !cfg.rules.revoke;
    }
    bool put_wait_on() const { return cfg.rules.put_wait && !xtra_own(); }
    // the 16 B local index is used with the log only (lease epochs need 32 bits; invalidate is not tested)
    bool li16_on() const { return cfg.li16 && log_scheme() && !cfg.rules.lease && !cfg.rules.revoke; }
    uint32_t gen_mask() const { return cfg.gen_bits >= 32 ? 0xFFFFFFFFu : (1u << cfg.gen_bits) - 1; }
    uint64_t gen_round() const { return (uint64_t)gen_mask() + 1; }
    uint64_t log_P_cycles() const { return (uint64_t)((double)cfg.log_poll_ns * tsc_per_ns()); }
    uint64_t log_W_ns() const { return cfg.log_wait_ns ? cfg.log_wait_ns : cfg.log_poll_ns; }
    uint64_t log_interval() const { return std::max<uint64_t>(cfg.log_interval_ns ? cfg.log_interval_ns : cfg.log_poll_ns / 2, 1000); }
    static inline uint64_t tsc_now() { _mm_lfence(); uint64_t t = __rdtsc(); _mm_lfence(); return t; }
    // the hit check's clock: rdtscp reads the TSC only after all earlier loads are globally visible
    static inline uint64_t tsc_after_loads() { unsigned aux; return __rdtscp(&aux); }
    // the next generation: allocation count mod 2^gen_bits, skipping 0 (= empty)
    uint32_t raw_gen(Host &X) {
        uint32_t m = gen_mask();
        do X.gcount++; while ((X.gcount & m) == 0);
        return (uint32_t)(X.gcount & m);
    }
    uint32_t peek_gen(Host &X) const {
        uint32_t m = gen_mask();
        uint64_t c = X.gcount + 1;
        while ((c & m) == 0) c++;
        return (uint32_t)(c & m);
    }
    void age_append(Host &X, Obj *ob) {
        ob->anext = nullptr;
        ob->aprev = X.age_tail;
        if (X.age_tail) X.age_tail->anext = ob; else X.age_head = ob;
        X.age_tail = ob;
    }
    void age_unlink(Host &X, Obj *ob) {
        if (!ob->aprev && X.age_head != ob) return;   // not linked
        if (ob->aprev) ob->aprev->anext = ob->anext; else X.age_head = ob->anext;
        if (ob->anext) ob->anext->aprev = ob->aprev; else X.age_tail = ob->aprev;
        ob->aprev = ob->anext = nullptr;
    }
    // log: publish the tail line {tail | epoch, allocation count} (mu held). Called after every allocation (before the
    // object or AI entry that uses it is written), so a reader that saw generation g in AI later reads count >= g here.
    void log_publish(int o) {
        Host &X = *hosts[o];
        TailEnt t{(X.log_tail & TAIL_MASK) | (uint64_t)X.log_epoch << 40, X.gcount};
        char *tl = owner_base(o) + off_log;
        store16v(tl, &t);
        wb_line(tl);
        _mm_sfence();
    }
    char *log_line(int o, uint64_t phys) const { return owner_base(o) + off_log + LINE + phys * LINE; }
    // log: append a record (mu held): write it with one 16 B store (and the lap header at a line start), write the line
    // back and wait for it (sfence), then advance the tail. w0 = DEL_W0: delete record; else a refresh record
    void log_append_rec(int o, uint64_t key, uint32_t w0, uint32_t gen) {
        if (xtra_own()) return;   // XTRA-style keeps no log (the tail line still carries the allocation count, R2)
        Host &X = *hosts[o];
        if (cfg.log_rec8) {   // compact lines (log_rec8): one slot per delete, two per refresh
            const bool two = w0 != DEL_W0;
            uint64_t i = X.log_tail;
            char *padl = nullptr;
            if (two && i % SPL == SPL - 1) {   // a refresh never straddles a line: pad the last slot
                padl = log_line(o, (i / SPL) % cfg.log_lines);
                vst64(padl + 8 + (SPL - 1) * 8, 0);
                i++;
            }
            uint64_t li = i / SPL, pos = i % SPL;
            char *l = log_line(o, li % cfg.log_lines);
            if (pos == 0) vst64(l, li / cfg.log_lines + 1);   // header: new lap number, empty refresh map
            vst64(l + 8 + pos * 8, key);
            if (two) {
                vst64(l + 8 + (pos + 1) * 8, (uint64_t)w0 << 32 | gen);
                vst64(l, vld64(l) | 1ULL << (56 + pos));
            }
            if (padl) wb_line(padl);
            wb_line(l);
            _mm_sfence();
            X.log_tail = i + (two ? 2 : 1);
            log_publish(o);
            X.st->log_records++;
            return;
        }
        uint64_t i = X.log_tail, li = i / RPL;
        char *l = log_line(o, li % cfg.log_lines);
        if (i % RPL == 0) vst64(l, li / cfg.log_lines + 1);
        LogRec r{key, w0, gen};
        store16v(l + 16 + (i % RPL) * 16, &r);
        wb_line(l);
        _mm_sfence();
        X.log_tail++;
        log_publish(o);
        X.st->log_records++;
    }
    void log_append(int o, uint64_t key) { log_append_rec(o, key, DEL_W0, 0); }
    bool feed_on() const { return log_scheme() && cfg.rules.feed && !xtra_own(); }
    // refresh record after a publish (mu held; the object and its AI entry are written back already)
    void log_refresh_rec(int o, uint64_t key, uint64_t off, int cls, uint32_t gen) {
        log_append_rec(o, key, (uint32_t)(off / LINE) | (uint32_t)cls << 28, gen);
    }
    // ---- kg table in CXL (Rules::kg_cxl): one u32 word per bucket of the owner's region, written only by the owner
    bool kg_cxl() const { return cfg.rules.kg_cxl && cfg.scheme != Scheme::PRIV && cfg.scheme != Scheme::SHARD; }
    uint64_t kg_bucket(uint64_t key) const { return mix64(key ^ 0x6a09e667f3bcc909ULL) & (cfg.kg_buckets - 1); }
    char *kg_word(int o, uint64_t b) const { return owner_base(o) + off_kg + b * 4; }
    // owner: one kg bump (mu held): the bucket word, written back (no wait), or the per-key map
    uint32_t kg_bump(int o, uint64_t key) {
        Host &X = *hosts[o];
        if (!kg_cxl()) return ++X.kg[key];
        uint64_t b = kg_bucket(key);
        uint32_t v = ++X.kgb[b];
        vst32(kg_word(o, b), v);
        wb_line(kg_word(o, b));
        return v;
    }
    // reader: the fill token = flush + load of the owner's bucket word
    uint32_t kg_load(int o, uint64_t key) {
        char *w = kg_word(o, kg_bucket(key));
        flush_line(w);
        _mm_mfence();
        tl_shm.rd++;
        return vld32(w);
    }
    // R2 restamp (mu held): rewrite a live object's lines with a new generation (same content), then its AI entry
    void restamp(int o, Obj *ob) {
        Host &X = *hosts[o];
        uint32_t g = raw_gen(X);
        if (log_scheme()) log_publish(o);
        char *base = slab(o) + ob->off;
        for (uint32_t i = 0; i < ob->nl; i++) {
            vst32(base + (size_t)i * LINE, g);
            wb_line(base + (size_t)i * LINE);
        }
        _mm_sfence();
        X.slot_gen[ob->off / LINE].store(g, std::memory_order_relaxed);
        if (ob->ai_idx != AI_NONE) {
            AiEnt cur = load16(&ai(o)[ob->ai_idx]);
            tl_shm.rd++;
            ai_write(o, ob->ai_idx, g, cur.kg, ob->key, ob->off, ob->cls);
            if (feed_on() && cfg.rules.feed_restamp) log_refresh_rec(o, ob->key, ob->off, ob->cls, g);
        }
        ob->gen = g;
        ob->gc = X.gcount;
        age_unlink(X, ob);
        age_append(X, ob);
        X.st->restamps++;
    }
    // allocate a generation (mu held): first restamp every live object that would otherwise be >= a quarter round old
    uint32_t new_gen(int o) {
        Host &X = *hosts[o];
        if (cfg.rules.restamp && cfg.scheme != Scheme::PRIV && cfg.scheme != Scheme::SHARD) {
            uint64_t A = gen_round() / 4;
            size_t n = X.objs.size() + 1;
            while (X.age_head && X.gcount + 1 - X.age_head->gc >= A && n--) restamp(o, X.age_head);
            if (X.age_head && X.gcount + 1 - X.age_head->gc >= A) X.st->restamp_overflow++;
        }
        uint32_t g = raw_gen(X);
        if (log_scheme()) log_publish(o);
        return g;
    }

    // ---------------------------------------------------------------- owner: SIEVE, allocator, admission
    void sieve_insert(Host &X, Obj *ob) {
        ob->prev = nullptr;
        ob->next = X.head;
        if (X.head) X.head->prev = ob;
        X.head = ob;
        if (!X.tail) X.tail = ob;
    }
    void sieve_unlink(Host &X, Obj *ob) {
        if (X.hand == ob) X.hand = ob->prev;
        if (ob->prev) ob->prev->next = ob->next; else X.head = ob->next;
        if (ob->next) ob->next->prev = ob->prev; else X.tail = ob->prev;
        ob->prev = ob->next = nullptr;
    }
    void free_slot_at(Host &X, uint64_t off, int cls) {
        X.slot_gen[off / LINE].store(0, std::memory_order_relaxed);
        X.visited[off / LINE].store(0, std::memory_order_relaxed);
        X.freel[cls].push_back(off);
    }
    void free_slot(Host &X, Obj *ob) { free_slot_at(X, ob->off, ob->cls); }
    int owner_n() const { return cfg.owners > 0 ? cfg.owners : H; }
    // Give an object's slot back (mu held; its charges were already removed). Immediate reuse (the design: readers
    // detect reuse by generation), or delayed reuse (Config::reuse_delayed, a PCCIndex-style baseline): the slot and its
    // charges stay taken until every other live host has dropped its entry, flushed the lines and acked.
    // charged: the caller kept the object's charges in place (delayed reuse, PUT replacing it)
    void release_slot(int o, uint64_t key, uint64_t off, int cls, uint32_t nl, uint64_t obj_b, uint64_t li_b,
                      bool charged = false) {
        Host &X = *hosts[o];
        if (!cfg.reuse_delayed) { free_slot_at(X, off, cls); return; }
        if (!charged) {
            X.used += obj_b + li_b;
            X.obj_bytes += obj_b;
            X.li_bytes += li_b;
        }
        X.pending_bytes += obj_b + li_b;
        reclaim_local(o, o, key, off, nl);   // the owner host's own readers
        uint32_t need = 0;
        for (int r = 0; r < H; r++)
            if (r != o && !(X.dead_mask >> r & 1)) need |= 1u << r;
        uint64_t id = ++X.reclaim_seq;
        X.reclaims[id] = {key, off, cls, need, obj_b, li_b, mono_ns()};
        if (!need) { reclaim_finish(o, id); return; }
        for (int r = 0; r < H; r++) {
            if (!(need >> r & 1)) continue;
            MsgHdr m{};
            m.type = M_RECL; m.nlines = 1; m.key = key; m.a = (uint32_t)(off / LINE); m.c = nl; m.e = id;
            send(o, r, &m, nullptr, 0, false);
        }
    }
    void reclaim_finish(int o, uint64_t id) {   // mu held
        Host &X = *hosts[o];
        auto it = X.reclaims.find(id);
        if (it == X.reclaims.end()) return;
        Host::Reclaim &c = it->second;
        free_slot_at(X, c.off, c.cls);
        X.used -= c.obj_bytes + c.li_bytes;
        X.pending_bytes -= c.obj_bytes + c.li_bytes;
        X.obj_bytes -= c.obj_bytes;
        X.li_bytes -= c.li_bytes;
        X.reclaims.erase(it);
        X.st->reclaim_done++;
    }
    // reader host r drops its entry for key if it names the slot at off of owner o, and flushes the slot's lines
    void reclaim_local(int r, int o, uint64_t key, uint64_t off, uint32_t nl) {
        Host &R = *hosts[r];
        uint64_t v;
        Stripe &s = stripe(r, key);
        std::lock_guard<std::mutex> g(s.mu);
        if (R.li.find(key, o, &v) && (v >> 32 & ((1ULL << 28) - 1)) * LINE == off) R.li.erase(key, o, v);
        obj_flush(r, o, off, nl);
    }
    // owner o's periodic work (handler thread, every ms): gauges for the time series, delayed-reuse timeouts
    void owner_tick(int o) {
        Host &X = *hosts[o];
        uint64_t now = mono_ns();
        if (now < X.next_gauge) return;
        X.next_gauge = now + 1'000'000;
        std::lock_guard<std::mutex> g(X.mu);
        now = mono_ns();   // after the lock: a reclaim created while we waited for it must not look 2^64 ns old
        if (cfg.reuse_delayed && cfg.reuse_timeout_ns) {
            std::vector<uint64_t> late;
            for (auto &kv : X.reclaims)
                if (now > kv.second.t && now - kv.second.t > cfg.reuse_timeout_ns) late.push_back(kv.first);
            for (uint64_t id : late) {
                X.dead_mask |= X.reclaims[id].need;   // declared dead: never waited for again (no fencing here)
                reclaim_finish(o, id);
                X.st->reclaim_timeout++;
            }
        }
        X.st->owner_used = X.used;
        X.st->owner_objects = X.objs.size();
        X.st->slab_bump = X.bump;
        uint64_t fb = 0;
        for (int c = 0; c < NCLS; c++) fb += X.freel[c].size() * CLS[c] * LINE;
        X.st->slab_free = fb;
        if (log_scheme()) {   // second item of the tail line: object count for the readers' local-index caps
            TailEnt t2{X.objs.size(), 0};
            store16v(owner_base(o) + off_log + 16, &t2);
            wb_line(owner_base(o) + off_log);
        }
        X.st->owner_limit = budget_eff;
        X.st->reclaim_pending = X.reclaims.size();
        X.st->dead_hosts = X.dead_mask | X.ack_dead.load(std::memory_order_relaxed);
    }
    // eviction: unpublish the AI entry and free the slot. No messages (readers detect reuse by gen).
    void evict_obj(int o, Obj *ob) {
        Host &X = *hosts[o];
        if (ob->ai_idx != AI_NONE) ai_delete(o, ob->ai_idx);
        if (xtra_own() && cfg.rules.xtra_kill_on_evict) {   // CD versions: after the AI delete is written back (it fences)
            xver_bump(o, ob->key);
            _mm_sfence();
        }
        sieve_unlink(X, ob);
        age_unlink(X, ob);
        uint64_t ob_b = ob->obj_bytes, li_b = ob->li_bytes;
        uncharge_obj(X, ob);
        release_slot(o, ob->key, ob->off, ob->cls, ob->nl, ob_b, li_b);
        X.objs.erase(ob->key);
        delete ob;
        X.st->evictions++;
    }
    bool evict_one(int o) {
        Host &X = *hosts[o];
        if (!X.tail || (X.head == X.tail && X.tail->pinned)) return false;
        Obj *h = X.hand ? X.hand : X.tail;
        while (h->pinned || X.visited[h->off / LINE].load(std::memory_order_relaxed)) {
            if (!h->pinned) X.visited[h->off / LINE].store(0, std::memory_order_relaxed);
            h = h->prev ? h->prev : X.tail;
        }
        X.hand = h->prev;
        evict_obj(o, h);
        return true;
    }
    // Write a new object and publish it (mu held). ai_hint: reuse this AI slot (PUT replacing the key).
    // replace: a pinned, already uncharged object whose SIEVE position the new one takes (write = access).
    bool install(int o, uint64_t key, uint32_t ver, const char *val, uint32_t vlen, uint32_t kgv, int64_t ai_hint,
                 Obj *replace = nullptr, bool refresh = true) {
        Host &X = *hosts[o];
        uint32_t nl = obj_lines(vlen);
        int cls = class_of(nl);
        if (cls < 0) return false;
        expire_ts_charge(X);
        const bool k11b = cfg.charge_model == CHARGE_K11B;
        uint64_t ob_bytes = cfg.charge_value_bytes_only ? vlen : k11b ? k11b_charge(vlen) : (uint64_t)CLS[cls] * LINE;
        uint64_t li_bytes = cfg.charge_value_bytes_only || (k11b && !k11b_host_entries()) ? 0 : (uint64_t)H * cfg.li_entry_bytes;
        uint64_t charge = ob_bytes + li_bytes;
        if (charge > budget_eff) return false;
        // delayed reuse: evicted objects stay charged until their reclaim is acked; count them as space on its way, so
        // one admission does not evict the whole cache before any ack can arrive, but admit only into space really free
        while (X.used - X.pending_bytes + charge > budget_eff)
            if (!evict_one(o)) return false;
        if (X.used + charge > budget_eff) { X.st->admit_wait_reclaim++; return false; }
        uint64_t off;
        for (int tries = 0;; tries++) {
            if (!X.freel[cls].empty()) { off = X.freel[cls].back(); X.freel[cls].pop_back(); break; }
            uint64_t cls_bytes = (uint64_t)CLS[cls] * LINE;   // slab space (the charge may differ: see Config)
            if (X.bump + cls_bytes <= cfg.slab_bytes) { off = X.bump; X.bump += cls_bytes; break; }
            if (tries >= 64 || !evict_one(o)) return false;   // fragmentation: give up after 64 evictions
            X.st->frag_evictions++;
        }
        uint64_t idx;
        if (ai_hint >= 0) idx = (uint64_t)ai_hint;
        else if (!ai_alloc(o, key, &idx)) { X.freel[cls].push_back(off); return false; }
        uint32_t gen = new_gen(o);
        obj_write(slab(o) + off, gen, key, val, vlen);
        X.slot_gen[off / LINE].store(gen, std::memory_order_relaxed);
        X.visited[off / LINE].store(0, std::memory_order_relaxed);
        ai_write(o, idx, gen, kgv, key, off, cls);   // publish
        X.ai_used[idx] = AI_FULL;
        if (feed_on() && refresh) log_refresh_rec(o, key, off, cls, gen);   // after the publish is written back
        Obj *ob = new Obj{key, off, idx, gen, ver, cls};
        ob->nl = nl;
        ob->gc = X.gcount;
        age_append(X, ob);
        ob->charge = charge;
        ob->obj_bytes = ob_bytes;
        ob->li_bytes = li_bytes;
        if (replace) {   // take the replaced object's place in the SIEVE list, visited
            ob->prev = replace->prev;
            ob->next = replace->next;
            if (ob->prev) ob->prev->next = ob; else X.head = ob;
            if (ob->next) ob->next->prev = ob; else X.tail = ob;
            if (X.hand == replace) X.hand = ob;
            replace->prev = replace->next = nullptr;
            X.visited[off / LINE].store(1, std::memory_order_relaxed);
        } else {
            sieve_insert(X, ob);
        }
        X.objs[key] = ob;
        X.used += charge;
        X.obj_bytes += ob_bytes;
        X.li_bytes += li_bytes;
        X.st->admits++;
        if (cfg.on_publish) cfg.on_publish(key, ver);
        return true;
    }
    void uncharge_obj(Host &X, Obj *ob) {
        X.used -= ob->charge;
        X.obj_bytes -= ob->obj_bytes;
        X.li_bytes -= ob->li_bytes;
        ob->charge = ob->obj_bytes = ob->li_bytes = 0;
    }
    // tombstones of finished PUTs stop being charged slow_limit_ns after the PUT started (mu held)
    void expire_ts_charge(Host &X) {
        uint64_t now = mono_ns(), c = (uint64_t)H * cfg.ts_entry_bytes;
        while (!X.ts_expiry.empty() && X.ts_expiry.front() <= now) {
            X.ts_expiry.pop_front();
            X.used -= c;
            X.ts_bytes -= c;
        }
    }
    uint32_t kg_of(Host &X, uint64_t key) {
        if (kg_cxl()) return X.kgb[kg_bucket(key)];
        auto it = X.kg.find(key);
        return it == X.kg.end() ? 0 : it->second;
    }
    // miss fill arriving at the owner (from a message or a local miss)
    void owner_fill(int o, uint64_t key, uint32_t tok, const char *val, uint32_t vlen, uint32_t ver) {
        Host &X = *hosts[o];
        if (!vlen) return;   // a deleted key's tombstone is never cached (the GET paths send no such fill)
        if (mono_ns() < X.writable_at.load()) { X.st->fill_drop_restart++; return; }
        std::lock_guard<std::mutex> g(X.mu);
        uint32_t cur = kg_of(X, key);
        if (cfg.rules.filltok && tok != cur) { X.st->fill_drop_tok++; return; }
        if (cfg.rules.pending && X.pending.count(key)) { X.st->fill_drop_pending++; return; }
        if (X.objs.count(key)) { X.st->fill_drop_cached++; return; }
        if (!install(o, key, ver, val, vlen, cur, -1)) X.st->fill_drop_space++;
    }
    // cs != nullptr: an owner's own client (count locally; the handler's heat messages count in Stats)
    void apply_heat(int o, uint64_t off, uint32_t gen, ClientStats *cs = nullptr) {
        Host &X = *hosts[o];
        if (off % LINE == 0 && off < cfg.slab_bytes &&
            X.slot_gen[off / LINE].load(std::memory_order_relaxed) == gen) {
            // set only if clear: threads hitting a hot object do not all write its line
            if (!X.visited[off / LINE].load(std::memory_order_relaxed)) X.visited[off / LINE].store(1, std::memory_order_relaxed);
            if (cs) cs->heat_applied++; else X.st->heat_applied++;
        } else {
            if (cs) cs->heat_stale++; else X.st->heat_stale++;
        }
    }

    // ---------------------------------------------------------------- revocation (any host)
    void do_revoke(int h, uint64_t key, uint32_t kgv) {
        Stripe &s = stripe(h, key);
        std::lock_guard<std::mutex> g(s.mu);
        hosts[h]->li.erase(key, Cache::owner_of_static(key, owner_n()));
        if (cfg.rules.tomb) {
            TsEnt &t = s.ts[key];   // a new entry starts as {0, 0}
            t.kg = std::max(t.kg, kgv);
            t.t = mono_ns();
        }
        hosts[h]->st->revokes++;
    }

    // ---------------------------------------------------------------- PUT (owner's executor thread)
    // unpublish the key's AI entry when a PUT starts (mu held)
    void put_unpublish(int o, uint64_t key) {
        Host &X = *hosts[o];
        auto it = X.objs.find(key);
        if (it != X.objs.end() && it->second->ai_idx != AI_NONE) {
            ai_delete(o, it->second->ai_idx);
            it->second->ai_idx = AI_NONE;
        }
    }
    // PUT. Log: (1) kg+1, unpublish the AI entry (2) append the record -- together the cut (3) backend, bump2,
    // optionally publish (4) the executor returns W after the cut. Lease: (1) is the cut, the executor returns T after.
    // op W_DEL / W_EXPIRE = the same steps without the publish (Rules del_*); W_DEL writes the backend's
    // tombstone (its version is the reply), W_EXPIRE does not touch the backend (reply 1, no version)
    uint32_t do_put(int o, uint64_t key, int op = W_PUT) {
        put_begin(o, key, op);
        char buf[MAX_OUT];
        uint32_t vlen = 0;
        uint32_t ver = be_write(key, buf, &vlen, op, false);
        return put_end(o, key, ver, buf, vlen, op);
    }
    // the backend side of a write op: PUT a new value, DELETE a tombstone (vlen 0), EXPIRE nothing (version 1 = done);
    // complete = the asynchronous executor applying a write it issued put_latency_ns earlier
    uint32_t be_write(uint64_t key, char *buf, uint32_t *vlen, int op, bool complete) {
        uint32_t v;
        if (op == W_EXPIRE) { *vlen = 0; v = 1; }
        else if (op == W_DEL) { *vlen = 0; v = complete ? be->del_complete(key) : be->del(key); }
        else v = complete ? be->put_complete(key, buf, vlen) : be->put(key, buf, vlen);
#if RC_MONO
        if (cfg.mono) tl_be_done_tsc = tsc_now();   // Config::mono: the swap's cut must follow within FD (mono_fd_ns)
#endif
        return v;
    }
    // a PUT up to the backend write: steps 1 and 2 (kg+1, unpublish, record, cut; invalidate: revoke + acks). Sets
    // tl_cut_ns / tl_cut_tsc / tl_ps. With put_async the asynchronous executor issues the write after this.
    void put_begin(int o, uint64_t key, int op = W_PUT) {
        Host &X = *hosts[o];
        const uint64_t start_tsc = tsc_now();   // M4 ablation log_wait_from_cut = false: W counts from here
        uint32_t nkg;
        const bool logm = log_scheme();
        const bool isput = op == W_PUT;   // DELETE / EXPIRE: the del_* ablations below
        const bool del = logm && cfg.rules.log_every_put && (!cfg.rules.feed || cfg.rules.feed_del) &&   // feed-nodel: none
                         (isput || cfg.rules.del_record);
        const bool unpub = isput || cfg.rules.del_unpublish;
        const bool rec_first = del && !cfg.rules.log_unpublish_first;   // ablation: record first (test G4-appendfirst)
        const bool rec_second = del && cfg.rules.log_unpublish_first;   // design
        if (swap_on_put()) {   // swap order: only `pending` here; the cut comes at the swap (put_end)
            {
                std::lock_guard<std::mutex> g(X.mu);
                if (cfg.rules.pending) X.pending.insert(key);
            }
            if (cfg.diag_put_timing) tl_ps.unpub = __rdtsc();
            hook(Hook::PutAfterStep1, o, key);
            hook(Hook::PutAfterBump1, o, key);
            hook(Hook::PutAfterAcks, o, key);
            return;
        }
        {
            std::lock_guard<std::mutex> g(X.mu);
            nkg = isput || cfg.rules.del_bump ? kg_bump(o, key) : kg_of(X, key);
            if (cfg.rules.pending) X.pending.insert(key);
            if (cfg.rules.revoke && !cfg.charge_value_bytes_only) {
                uint64_t c = (uint64_t)H * cfg.ts_entry_bytes;   // the tombstones this PUT leaves on every host
                X.used += c;
                X.ts_bytes += c;
                X.ts_expiry.push_back(mono_ns() + cfg.slow_limit_ns);
            }
            if (rec_first) log_append(o, key);
            else if (cfg.rules.unpublish_on_put) {   // no stale AI entry is readable after this point (the cut)
                if (unpub) put_unpublish(o, key);
                tl_cut_ns = mono_ns();
            }
        }
        if (cfg.diag_put_timing) tl_ps.unpub = __rdtsc();
        hook(Hook::PutAfterStep1, o, key);
        if (logm) {
            std::lock_guard<std::mutex> g(X.mu);
            if (rec_first) {
                if (cfg.rules.unpublish_on_put && unpub) put_unpublish(o, key);
            } else if (rec_second) {
                log_append(o, key);
            }
            _mm_mfence();
            tl_cut_tsc = tsc_now();   // the cut: the record is visible (written back) and the AI entry is gone
            tl_cut_mono = mono_ns();
            tl_ps.cut = tl_cut_tsc;
            if (!cfg.rules.log_wait_from_cut) tl_cut_tsc = start_tsc;   // ablation (M4): W from the start (test M4-waitfromcut)
        }
        hook(Hook::PutAfterBump1, o, key);
        if (cfg.rules.revoke && (isput || cfg.rules.del_record)) {   // invalidate: revoke everywhere, wait for the acks
            Host::Lane &L = *X.lanes[tl_lane];
            uint64_t putid = ++L.putseq << 8 | (uint64_t)tl_lane;
            uint32_t dead = X.ack_dead.load(std::memory_order_relaxed), need = 0;
            for (int r = 0; r < H; r++)
                if (r != o && !(dead >> r & 1)) need |= 1u << r;
            L.acked.store(0);
            L.acks_needed.store(__builtin_popcount(need));
            L.cur_put.store(putid);
            do_revoke(o, key, nkg);
            for (int r = 0; r < H; r++) {
                if (!(need >> r & 1)) continue;   // hosts declared dead (revoke_timeout_ns) are not revoked any more
                MsgHdr m{};
                m.type = M_REVOKE; m.nlines = 1; m.key = key; m.b = nkg; m.e = putid;
                send(o, r, &m, nullptr, 0, false);
            }
            uint64_t t0 = mono_ns();
            while (L.acks_needed.load() > 0 && !stop.load()) {
                _mm_pause();
                // revoke_timeout_ns > 0: hosts that did not ack in time are declared dead (no fencing): never waited
                // for again. A host that wakes up keeps the entries it was never told to drop.
                if (cfg.revoke_timeout_ns && mono_ns() - t0 > cfg.revoke_timeout_ns) {
                    uint32_t late = need & ~L.acked.load();
                    X.ack_dead.fetch_or(late);
                    X.st->revoke_dead_declared += __builtin_popcount(late);
                    X.st->dead_hosts = X.ack_dead.load();
                    break;
                }
            }
        }
        hook(Hook::PutAfterAcks, o, key);   // (lease: right after the first kg bump / unpublish)
    }
    // after the backend write (version ver, value buf): step 3 (bump2, publish the new object)
    uint32_t put_end(int o, uint64_t key, uint32_t ver, const char *buf, uint32_t vlen, int op = W_PUT) {
        Host &X = *hosts[o];
        const bool logm = log_scheme();
        const bool isput = op == W_PUT;   // DELETE / EXPIRE publish nothing: the old object goes, no refresh record
        if (cfg.diag_put_timing) tl_ps.backend = __rdtsc();
        hook(Hook::PutAfterBackend, o, key);
        {
            std::lock_guard<std::mutex> g(X.mu);
            const bool swap = swap_on_put();
            // swap: the delete record is the cut, W (put_wait) counts from it.
            // Design order: kg+1, the new object replaces the old one (AI entry switched), the
            // delete record = the cut, then (swap_refresh_after_del) the new object's refresh record.
            const bool swap_del = swap && logm && cfg.rules.log_every_put && (!cfg.rules.feed || cfg.rules.feed_del) &&
                                  (isput || cfg.rules.del_record);
            const bool late_ref = swap && !cfg.rules.swap_del_first && cfg.rules.swap_refresh_after_del && feed_on();
            auto swap_cut = [&] {
                if (swap_del) log_append(o, key);
                _mm_mfence();
                tl_cut_ns = mono_ns();
                tl_cut_mono = tl_cut_ns;
                tl_cut_tsc = tsc_now();
#if RC_MONO
                if (cfg.mono && tl_be_done_tsc && (double)(tl_cut_tsc - tl_be_done_tsc) > (double)cfg.mono_fd_ns * tsc_per_ns())
                    X.st->mono_fd_overrun++;   // Config::mono: the table's FD + P would be too short for this PUT
#endif
                tl_ps.cut = tl_cut_tsc;
                // the version is known: report the cut right here (EXPIRE makes no version)
                if (cfg.on_cut && op != W_EXPIRE) cfg.on_cut(key, ver, tl_cut_mono);
            };
            // swap: the only bump (fills taken before it are dropped)
            if ((swap || cfg.rules.bump2) && (isput || cfg.rules.del_bump)) kg_bump(o, key);
            if (swap && cfg.rules.swap_del_first) {   // ablation: the cut before the switch (test C2-swaporder)
                swap_cut();
                hook(Hook::PutSwapMid, o, key);
            }
            uint32_t cur = kg_of(X, key);
            auto it = X.objs.find(key);
            Obj *old = it == X.objs.end() ? nullptr : it->second;
            uint64_t old_ob = 0, old_li = 0;
            // xtra_own (CD versions): install() moves the key's slot version when it publishes; the PUT moves it once
            // more after the switch (below), which also covers a PUT that does not publish. Ablation: no bump in this PUT.
            tl_xbump_off = xtra_own() && !cfg.rules.xtra_kill_on_put;
            tl_xbump_key = key;
            if (!isput && !cfg.rules.del_unpublish) old = nullptr;   // ablation: the deleted object stays cached and published
            if (op == W_EXPIRE && old) X.st->expire_dropped++;
            if (old) {   // keep its slot (and AI entry, if still published) until the new object is published
                X.objs.erase(key);
                age_unlink(X, old);
                old_ob = old->obj_bytes; old_li = old->li_bytes;
                // delayed reuse: the old object's memory is not reusable until reclaimed, so it stays charged while the
                // new one is admitted (immediate reuse: its slot is freed right after, so its charge goes now)
                if (!cfg.reuse_delayed) uncharge_obj(X, old);
                if (cfg.write_is_access) old->pinned = true;   // keeps its SIEVE position for the new object
                else sieve_unlink(X, old);
            }
            bool ok = isput && cfg.rules.put_publish &&
                      install(o, key, ver, buf, vlen, cur, old && old->ai_idx != AI_NONE ? (int64_t)old->ai_idx : -1,
                              old && cfg.write_is_access ? old : nullptr, !late_ref);
            if (old) {
                if (!ok && old->pinned) sieve_unlink(X, old);
                if (!ok && old->ai_idx != AI_NONE) ai_delete(o, old->ai_idx);
                release_slot(o, key, old->off, old->cls, old->nl, old_ob, old_li, cfg.reuse_delayed);
                delete old;
            }
            if (!ok && isput) X.st->put_nocache++;
            if (xtra_own()) xver_bump(o, key);   // after the switch is written back; swap_cut's mfence orders it before the reply
            tl_xbump_off = false;
            if (swap && !cfg.rules.swap_del_first) {   // the switch is written back (install / ai_delete); now the cut
                hook(Hook::PutSwapMid, o, key);
                swap_cut();
                if (late_ref && ok) {   // after the delete record: holders end repointed to the new object, not deleted
                    const Obj *nb = X.objs[key];
                    log_refresh_rec(o, key, nb->off, nb->cls, nb->gen);
                }
            } else if (!swap && !cfg.rules.unpublish_on_put) tl_cut_ns = mono_ns();   // the old entry was replaced / deleted now
            if (logm && !cfg.rules.log_every_put) {   // ablation: a record only when publishing (test G5-atpublish)
                if (ok) log_append(o, key);
                _mm_mfence();
                tl_cut_tsc = tsc_now();
                tl_cut_mono = mono_ns();
            }
            X.pending.erase(key);
            // window order: the version exists now
            if (logm && !swap && cfg.on_cut && op != W_EXPIRE) cfg.on_cut(key, ver, tl_cut_mono);
        }
        (isput ? X.st->puts : op == W_DEL ? X.st->deletes : X.st->expires)++;
        return ver;
    }

    // ---------------------------------------------------------------- PRIV baseline (host-local private caches)
    uint64_t priv_charge(uint32_t vlen) const {
        if (cfg.charge_model == CHARGE_K11B && !cfg.charge_value_bytes_only) return k11b_charge(vlen);
        return cfg.charge_value_bytes_only ? vlen : (uint64_t)vlen + HDR + cfg.li_entry_bytes;   // value + header + index entry
    }
    uint64_t k11b_charge(uint32_t v) const { return charge_k11b(cfg.scheme, v); }
    // CHARGE_K11B charges with host parts: Krater (N x local entry) and MEGALON-style (N x index replica)
    bool k11b_host_entries() const {
        return cfg.scheme == Scheme::RACKCACHE || cfg.scheme == Scheme::GSTATE_CXL || cfg.scheme == Scheme::GSTATE_COH;
    }
    void priv_unlink(Priv &V, PObj *ob) {
        if (V.hand == ob) V.hand = ob->prev;
        if (ob->prev) ob->prev->next = ob->next; else V.head = ob->next;
        if (ob->next) ob->next->prev = ob->prev; else V.tail = ob->prev;
        ob->prev = ob->next = nullptr;
    }
    void priv_drop(Host &X, PObj *ob) {   // mu held: unindex, free the slot (readers detect reuse by gen)
        Priv &V = X.priv;
        V.index.erase(ob->key);
        priv_unlink(V, ob);
        V.visited[ob->off / LINE].store(0, std::memory_order_relaxed);
        V.freel[ob->cls].push_back(ob->off);
        V.used -= ob->charge;
        V.objs.erase(ob->key);
        delete ob;
    }
    bool priv_evict_one(Host &X) {
        Priv &V = X.priv;
        if (!V.tail) return false;
        PObj *h = V.hand ? V.hand : V.tail;
        while (V.visited[h->off / LINE].load(std::memory_order_relaxed)) {
            V.visited[h->off / LINE].store(0, std::memory_order_relaxed);
            h = h->prev ? h->prev : V.tail;
        }
        V.hand = h->prev;
        priv_drop(X, h);
        X.st->evictions++;
        return true;
    }
    // lock-free read: index (seqlock) -> one 64 B load per line, gen checked, key in line 0
    bool priv_get(int h, uint64_t key, char *out, uint32_t *vlen) {
        Priv &V = hosts[h]->priv;
        uint64_t v;
        if (!V.index.find(key, &v)) return false;
        uint64_t off = (v >> 32 & ((1ULL << 28) - 1)) * LINE;
        uint32_t gen = (uint32_t)v, maxl = CLS[v >> 60];
        if (off + LINE > V.slab_bytes) return false;
        alignas(64) char l[LINE];
        load64(V.slab + off, l);
        uint32_t g, vl;
        uint64_t k;
        memcpy(&g, l, 4); memcpy(&k, l + 4, 8); memcpy(&vl, l + 12, 4);
        if (g != gen || k != key || vl > MAX_VLEN || obj_lines(vl) > maxl) return false;
        uint32_t done = std::min<uint32_t>(vl, PAY - HDR);
        memcpy(out, l + 4 + HDR, done);
        for (uint32_t i = 1, nl = obj_lines(vl); i < nl; i++) {
            load64(V.slab + off + (uint64_t)i * LINE, l);
            memcpy(&g, l, 4);
            if (g != gen) return false;
            uint32_t n = std::min<uint32_t>(vl - done, PAY);
            memcpy(out + done, l + 4, n);
            done += n;
        }
        *vlen = vl;
        if (!V.visited[off / LINE].load(std::memory_order_relaxed)) V.visited[off / LINE].store(1, std::memory_order_relaxed);
        return true;
    }
    // shard_direct: SHARD's kg also kept as lock-free bucket words, so a direct remote miss reads its token without the
    // owner's mutex (keys sharing a bucket only refuse more fills, as with the kg table in CXL, kg_cxl)
    std::atomic<uint32_t> &priv_kgb(Host &X, uint64_t key) { return X.priv.kgb[kg_bucket(key)]; }
    uint32_t priv_token(int h, uint64_t key) {
        if (cfg.shard_direct) return priv_kgb(*hosts[h], key).load(std::memory_order_acquire);
        Priv &V = hosts[h]->priv;
        std::lock_guard<std::mutex> g(V.mu);
        auto it = V.kg.find(key);
        return it == V.kg.end() ? 0 : it->second;
    }
    void priv_fill(int h, uint64_t key, uint32_t tok, const char *val, uint32_t vlen, uint32_t ver) {
        std::lock_guard<std::mutex> g(hosts[h]->priv.mu);
        priv_fill_locked(h, key, tok, val, vlen, ver);
    }
    void priv_fill_locked(int h, uint64_t key, uint32_t tok, const char *val, uint32_t vlen, uint32_t ver) {   // V.mu held
        Host &X = *hosts[h];
        Priv &V = X.priv;
        if (!vlen) return;   // a deleted key's tombstone is never cached
        auto k = V.kg.find(key);
        uint32_t cur = cfg.shard_direct ? priv_kgb(X, key).load(std::memory_order_relaxed) : k == V.kg.end() ? 0 : k->second;
        if (cfg.rules.filltok && tok != cur) { X.st->fill_drop_tok++; return; }
        if (cfg.rules.pending && V.pending.count(key)) { X.st->fill_drop_pending++; return; }
        if (V.objs.count(key)) { X.st->fill_drop_cached++; return; }
        int cls = class_of(obj_lines(vlen));
        uint64_t charge = priv_charge(vlen);
        if (cls < 0 || charge > cfg.budget_bytes) { X.st->fill_drop_space++; return; }
        while (V.used + charge > cfg.budget_bytes)
            if (!priv_evict_one(X)) { X.st->fill_drop_space++; return; }
        uint64_t cls_bytes = (uint64_t)CLS[cls] * LINE, off;
        for (int tries = 0;; tries++) {
            if (!V.freel[cls].empty()) { off = V.freel[cls].back(); V.freel[cls].pop_back(); break; }
            if (V.bump + cls_bytes <= V.slab_bytes) { off = V.bump; V.bump += cls_bytes; break; }
            if (tries >= 64 || !priv_evict_one(X)) { X.st->fill_drop_space++; return; }
            X.st->frag_evictions++;
        }
        uint32_t gen = ++V.genc;
        if (gen == 0) gen = ++V.genc;
        obj_write(V.slab + off, gen, key, val, vlen);   // gen first in every line, as in the shared area
        PObj *ob = new PObj{key, off, charge, gen, cls};
        ob->next = V.head;
        if (V.head) V.head->prev = ob;
        V.head = ob;
        if (!V.tail) V.tail = ob;
        V.objs[key] = ob;
        V.used += charge;
        V.visited[off / LINE].store(0, std::memory_order_relaxed);
        if (!V.index.put(key, li_pack(off, gen, cls))) {   // index full near this key: do not cache it
            V.objs.erase(key); priv_unlink(V, ob); V.freel[cls].push_back(off); V.used -= charge; delete ob;
            X.st->fill_drop_space++;
            return;
        }
        X.st->admits++;
        if (cfg.on_publish) cfg.on_publish(key, ver);
    }
    void priv_invalidate(int h, uint64_t key, int op = W_PUT) {   // PUT round 1 on host h (op: to count EXPIRE drops)
        Host &X = *hosts[h];
        Priv &V = X.priv;
        std::lock_guard<std::mutex> g(V.mu);
        ++V.kg[key];
        if (cfg.shard_direct) priv_kgb(X, key).fetch_add(1, std::memory_order_acq_rel);
        if (cfg.rules.pending) V.pending.insert(key);
        auto it = V.objs.find(key);
        if (it != V.objs.end()) {
            priv_drop(X, it->second);
            if (op == W_EXPIRE) X.st->expire_dropped++;
        }
        X.st->priv_inval++;
    }
    void priv_done(int h, uint64_t key) {   // PUT round 2 on host h
        Priv &V = hosts[h]->priv;
        std::lock_guard<std::mutex> g(V.mu);
        if (cfg.rules.bump2) ++V.kg[key];
        if (cfg.rules.bump2 && cfg.shard_direct) priv_kgb(*hosts[h], key).fetch_add(1, std::memory_order_acq_rel);
        V.pending.erase(key);
    }
    // PRIV PUT, run by the key's home host's executor: invalidate everywhere (acked), write the backend, then
    // tell every host the PUT is done. Round-2 messages travel on the same rings as the later PUTDONE, so the
    // writer has applied round 2 before it learns that its PUT returned.
    // DELETE / EXPIRE take both rounds too (the writer installs nothing); del_record off (ablation): no round 1,
    // the other hosts keep their copies
    uint32_t do_put_priv(int o, uint64_t key, int op = W_PUT) {
        Host &X = *hosts[o];
        Host::Lane &L = *X.lanes[tl_lane];
        if (op == W_PUT || cfg.rules.del_record) {
            uint64_t putid = ++L.putseq << 8 | (uint64_t)tl_lane;
            L.acks_needed.store(H - 1);
            L.cur_put.store(putid);
            priv_invalidate(o, key, op);
            for (int r = 0; r < H; r++) {
                if (r == o) continue;
                MsgHdr m{};
                m.type = M_PINV; m.nlines = 1; m.key = key; m.c = (uint32_t)op; m.e = putid;
                send(o, r, &m, nullptr, 0, false);
            }
            while (L.acks_needed.load() > 0 && !stop.load()) _mm_pause();
        }
        if (cfg.diag_put_timing) tl_ps.unpub = tl_ps.cut = __rdtsc();   // diag_put: "unpublish" = round 1 (invalidate + acks)
        hook(Hook::PutAfterAcks, o, key);
        char buf[MAX_OUT];
        uint32_t vlen;
        uint32_t ver = be_write(key, buf, &vlen, op, false);
        if (cfg.diag_put_timing) tl_ps.backend = __rdtsc();
        hook(Hook::PutAfterBackend, o, key);
        priv_done(o, key);
        for (int r = 0; r < H; r++) {
            if (r == o) continue;
            MsgHdr m{};
            m.type = M_PDONE; m.nlines = 1; m.key = key;
            send(o, r, &m, nullptr, 0, false);
        }
        (op == W_PUT ? X.st->puts : op == W_DEL ? X.st->deletes : X.st->expires)++;
        return ver;
    }

    // SHARD PUT at the owner (the only copy): invalidate, write the backend, done, install the new value
    // DELETE / EXPIRE = the same without installing anything (the owner's copy is the only one)
    uint32_t do_put_shard(int o, uint64_t key, int op = W_PUT) {
        Host &X = *hosts[o];
        if (cfg.rules.put_cut_at_publish) return do_put_shard_swap(o, key, op);
        priv_invalidate(o, key, op);
        if (cfg.diag_put_timing) tl_ps.unpub = tl_ps.cut = __rdtsc();   // diag_put: "unpublish" = the local invalidate
        hook(Hook::PutAfterAcks, o, key);
        char buf[MAX_OUT];
        uint32_t vlen;
        uint32_t ver = be_write(key, buf, &vlen, op, false);
        if (cfg.diag_put_timing) tl_ps.backend = __rdtsc();
        hook(Hook::PutAfterBackend, o, key);
        priv_done(o, key);
        if (op == W_PUT) priv_fill(o, key, priv_token(o, key), buf, vlen, ver);   // PUTs on a key are serialized here
        (op == W_PUT ? X.st->puts : op == W_DEL ? X.st->deletes : X.st->expires)++;
        return ver;
    }

    // Rules::put_cut_at_publish: SHARD's own order -- the owner keeps serving the old object during
    // the backend write (fills of the key are dropped: pending); after it, under the shard's lock: kg+1, drop the old
    // object, install the new one with the new token. No window: a reader sees the old or the new object.
    uint32_t do_put_shard_swap(int o, uint64_t key, int op = W_PUT) {
        Host &X = *hosts[o];
        Priv &V = X.priv;
        { std::lock_guard<std::mutex> g(V.mu); if (cfg.rules.pending) V.pending.insert(key); }
        if (cfg.diag_put_timing) tl_ps.unpub = __rdtsc();
        hook(Hook::PutAfterAcks, o, key);
        char buf[MAX_OUT];
        uint32_t vlen;
        uint32_t ver = be_write(key, buf, &vlen, op, false);
        if (cfg.diag_put_timing) tl_ps.backend = __rdtsc();
        hook(Hook::PutAfterBackend, o, key);
        {
            std::lock_guard<std::mutex> g(V.mu);
            ++V.kg[key];
            if (cfg.shard_direct) priv_kgb(X, key).fetch_add(1, std::memory_order_acq_rel);
            auto it = V.objs.find(key);
            if (it != V.objs.end()) {
                priv_drop(X, it->second);
                if (op == W_EXPIRE) X.st->expire_dropped++;
            }
            V.pending.erase(key);
            auto k = V.kg.find(key);
            uint32_t cur = cfg.shard_direct ? priv_kgb(X, key).load(std::memory_order_relaxed) : k == V.kg.end() ? 0 : k->second;
            if (op == W_PUT) priv_fill_locked(o, key, cur, buf, vlen, ver);
        }
        if (cfg.diag_put_timing) tl_ps.cut = __rdtsc();
        (op == W_PUT ? X.st->puts : op == W_DEL ? X.st->deletes : X.st->expires)++;
        return ver;
    }

    // ---------------------------------------------------------------- control channel
    uint64_t read_tail(int s, int r) {   // consumer progress of ring s->r, stored in r's area
        char *t = tail_line(r, s);
        flush_line(t);
        _mm_mfence();
        return vld64(t);
    }
    // try to append one encoded message (n lines) to ring s->r; out_mu[r] of host s held
    bool ring_put(int s, int r, const char *lines, uint32_t n) {
        Host &X = *hosts[s];
        uint64_t cap = cfg.ring_lines, head = X.head_local[r];
        uint64_t pad = ((head & (cap - 1)) + n > cap) ? cap - (head & (cap - 1)) : 0;
        if (head + pad + n - X.tail_cache[r] > cap) {
            X.tail_cache[r] = read_tail(s, r);
            if (head + pad + n - X.tail_cache[r] > cap) return false;
        }
        if (pad) {
            MsgHdr *p = (MsgHdr *)ring_line(s, r, head);
            MsgHdr m{};
            m.type = M_PAD; m.nlines = (uint32_t)pad;
            memcpy((void *)p, &m, LINE);
            wb_line(p);
            head += pad;
        }
        for (uint32_t i = 0; i < n; i++) {
            char *l = ring_line(s, r, head + i);
            memcpy(l, lines + (size_t)i * LINE, LINE);
            wb_line(l);
        }
        _mm_sfence();
        head += n;
        X.head_local[r] = head;
        char *hl = ring_hdr(s, r);
        vst64(hl, head);
        wb_line(hl);
        _mm_sfence();
        return true;
    }
    // Send a message from host s to host r. Droppable messages (fills, heat) are dropped when the ring is
    // full; the others go to a host-local overflow queue that s's handler drains (never blocks a handler).
    // ring traffic is not counted as shared-memory accesses of the operation: restore the counters, count one message
    struct ShmChannel {
        ShmCount keep = tl_shm;
        bool msg;
        explicit ShmChannel(bool m) : msg(m) {}
        ~ShmChannel() { tl_shm = keep; if (msg) tl_shm.msg++; }
    };
    void send(int s, int r, const MsgHdr *hdr, const void *data, uint32_t data_bytes, bool droppable) {
        ShmChannel ch(true);
        Host &X = *hosts[s];
        uint32_t n = 1 + (data_bytes + LINE - 1) / LINE;
        std::vector<uint8_t> enc((size_t)n * LINE, 0);
        memcpy(enc.data(), hdr, LINE);
        ((MsgHdr *)enc.data())->nlines = n;
        if (data_bytes) memcpy(enc.data() + LINE, data, data_bytes);
        std::unique_lock<std::mutex> g(*X.out_mu[r]);
        if (X.overflow[r].empty() && ring_put(s, r, (const char *)enc.data(), n)) { X.st->msgs_sent++; return; }
        if (droppable) {
            X.st->drop_ring_full++;
            g.unlock();
            if (hdr->type == M_FILL || hdr->type == M_SFILL) hook(Hook::FillDone, s, hdr->key);
            return;
        }
        X.overflow[r].push_back(std::move(enc));
        X.overflow_n++;
        X.st->overflowed++;
    }
    void flush_overflow(int s) {
        ShmChannel ch(false);
        Host &X = *hosts[s];
        if (X.overflow_n.load(std::memory_order_relaxed) == 0) return;
        for (int r = 0; r < H; r++) {
            std::lock_guard<std::mutex> g(*X.out_mu[r]);
            while (!X.overflow[r].empty()) {
                auto &m = X.overflow[r].front();
                if (!ring_put(s, r, (const char *)m.data(), (uint32_t)(m.size() / LINE))) break;
                X.overflow[r].pop_front();
                X.overflow_n--;
                X.st->msgs_sent++;
            }
        }
    }
    void deliver(int h, int reqid, uint32_t val, uint32_t val2 = 0, uint32_t seq = 0) {
        Host &X = *hosts[h];
        Mailbox *mb;
        {
            std::lock_guard<std::mutex> g(X.mb_mu);
            mb = X.mbox[reqid].get();
        }
        if (seq && seq != mb->seq.load()) return;   // late reply to a request that already timed out
        mb->val = val;
        mb->val2 = val2;
        mb->stamp = tl_reply_stamp;
        mb->ready.store(1, std::memory_order_release);
    }
    void process(int r, int s, const MsgHdr &m, const char *data) {
        Host &X = *hosts[r];
        X.st->msgs_recv++;
        switch (m.type) {
        case M_TOKREQ: {
            uint32_t tok;
            {
                std::lock_guard<std::mutex> g(X.mu);
                tok = kg_of(X, m.key);
            }
            MsgHdr o{};
            o.type = M_TOK; o.nlines = 1; o.key = m.key; o.a = m.a; o.b = tok; o.d = m.d;
            send(r, s, &o, nullptr, 0, false);
            break;
        }
        case M_TOK: deliver(r, m.a, m.b, 0, m.d); break;   // d = request seq: a late reply (after a timeout) is dropped
        case M_RECL: {   // delayed reuse: drop the entry for this slot, flush its lines, ack
            reclaim_local(r, s, m.key, (uint64_t)m.a * LINE, m.c);
            MsgHdr o{};
            o.type = M_RACK; o.nlines = 1; o.key = m.key; o.e = m.e;
            send(r, s, &o, nullptr, 0, false);
            break;
        }
        case M_RACK: {
            std::lock_guard<std::mutex> g(X.mu);
            auto it = X.reclaims.find(m.e);
            if (it != X.reclaims.end() && (it->second.need &= ~(1u << s)) == 0) reclaim_finish(r, m.e);
            break;
        }
        case M_FILL: {
            ShmCount s0 = tl_shm;
            owner_fill(r, m.key, m.a, data, m.b, m.c);
            shm_add(X.st->shm[SHM_FILL], s0);
            hook(Hook::FillDone, s, m.key);
            break;
        }
        case M_REVOKE: {
            do_revoke(r, m.key, m.b);
            MsgHdr o{};
            o.type = M_ACK; o.nlines = 1; o.key = m.key; o.e = m.e;
            send(r, s, &o, nullptr, 0, false);
            break;
        }
        case M_ACK:
            if ((m.e & 0xff) < X.lanes.size()) {
                Host::Lane &L = *X.lanes[m.e & 0xff];
                if (m.e == L.cur_put.load()) { L.acked.fetch_or(1u << s); L.acks_needed.fetch_sub(1); }
            }
            break;
        case M_PUTREQ: {
            Host::Lane &L = lane_of(X, m.key);
            std::lock_guard<std::mutex> g(L.ex_mu);
            L.jobs.push_back({m.key, s, (int)m.a, m.d, m.e, cfg.diag_put_timing ? __rdtsc() : 0, (int)m.c});
            L.njobs.fetch_add(1, std::memory_order_release);
            L.ex_cv.notify_one();
            break;
        }
        case M_PUTDONE:   // PRIV: the writer's kg right after this PUT's round 2 (same ring, nothing in between)
            tl_reply_stamp = m.e;
            deliver(r, m.a, m.b, cfg.scheme == Scheme::PRIV ? priv_token(r, m.key) : 0, m.d);
            break;
        case M_PINV: {
            priv_invalidate(r, m.key, (int)m.c);   // c: the WriteOp (EXPIRE drops are counted)
            MsgHdr o{};
            o.type = M_ACK; o.nlines = 1; o.key = m.key; o.e = m.e;
            send(r, s, &o, nullptr, 0, false);
            break;
        }
        case M_PDONE: priv_done(r, m.key); break;
        case M_SGET: {   // SHARD: owner serves a GET from its private cache
            static thread_local std::vector<char> v(MAX_OUT);
            uint32_t vlen;
            MsgHdr o{};
            o.nlines = 1; o.key = m.key; o.a = m.a;
            if (priv_get(r, m.key, v.data(), &vlen)) {
                o.type = M_SVAL; o.b = vlen;
                send(r, s, &o, v.data(), vlen, false);
            } else {
                o.type = M_SMISS; o.b = priv_token(r, m.key);
                send(r, s, &o, nullptr, 0, false);
            }
            break;
        }
        case M_SVAL: {
            Host &Y = *hosts[r];
            Mailbox *mb;
            { std::lock_guard<std::mutex> g(Y.mb_mu); mb = Y.mbox[m.a].get(); }
            memcpy(mb->buf, data, m.b);
            deliver(r, m.a, m.b, 1);
            break;
        }
        case M_SMISS: deliver(r, m.a, m.b, 0); break;
        case M_SFILL: priv_fill(r, m.key, m.a, data, m.b, m.c); hook(Hook::FillDone, s, m.key); break;
        case M_HEAT: {
            const HeatEnt *e = (const HeatEnt *)data;
            for (uint32_t i = 0; i < m.a; i++) apply_heat(r, e[i].off, e[i].gen);
            break;
        }
        default: fprintf(stderr, "bad message type %u\n", m.type); abort();
        }
    }
    // drain ring s->r on the receiver r; returns true if anything was processed
    bool drain(int s, int r) {
        Host &X = *hosts[r];
        char *hl = ring_hdr(s, r);
        flush_line(hl);
        _mm_mfence();
        uint64_t head = vld64(hl), t = X.rx_tail[s];
        if (t == head) return false;
        static thread_local std::vector<char> data;
        while (t < head) {
            const char *l = ring_line(s, r, t);
            flush_line(l);
            _mm_mfence();
            MsgHdr m;
            memcpy(&m, l, LINE);
            if (m.type == M_PAD) { t += m.nlines; continue; }
            data.resize((size_t)m.nlines * LINE);
            for (uint32_t i = 1; i < m.nlines; i++) flush_line(ring_line(s, r, t + i));
            _mm_mfence();
            for (uint32_t i = 1; i < m.nlines; i++) memcpy(data.data() + (size_t)(i - 1) * LINE, ring_line(s, r, t + i), LINE);
            process(r, s, m, data.data());
            t += m.nlines;
        }
        X.rx_tail[s] = t;
        char *tl = tail_line(r, s);
        vst64(tl, t);
        wb_line(tl);
        _mm_sfence();
        return true;
    }
    void bind(int h) {
        if (!cfg.bind_hosts || numa_available() < 0) return;
        numa_run_on_node(h % (numa_max_node() + 1));
    }
    // reclaim tombstones older than slow_limit_ns: one stripe per call, a full pass every slow_limit_ns
    void sweep_tombstones(int h) {
        Host &X = *hosts[h];
        uint64_t now = mono_ns();
        if (now < X.next_sweep) return;
        X.next_sweep = now + std::max<uint64_t>(cfg.slow_limit_ns / NSTRIPE, 1000);
        Stripe &s = X.stripes[X.sweep_pos++ % NSTRIPE];
        std::lock_guard<std::mutex> g(s.mu);
        for (auto it = s.ts.begin(); it != s.ts.end();)
            if (now - it->second.t > cfg.slow_limit_ns) { it = s.ts.erase(it); X.st->ts_reclaimed++; }
            else ++it;
    }
    void handler_loop(int h) {
        bind(h);
        if (cfg.exec_timer_slack_ns) prctl(PR_SET_TIMERSLACK, (unsigned long)cfg.exec_timer_slack_ns, 0, 0, 0);
        int idle = 0;
        Host &X = *hosts[h];
        while (!stop.load(std::memory_order_relaxed)) {
            bool any = false;
            uint64_t c0 = __rdtsc();
            if (X.unreachable.load(std::memory_order_relaxed)) { usleep(100); continue; }   // test: partitioned
            for (int s = 0; s < H; s++)
                if (s != h) any |= drain(s, h);
            if (any) X.st->handler_busy_cycles += __rdtsc() - c0;
            flush_overflow(h);
            sweep_tombstones(h);
            if (h < owner_n()) owner_tick(h);
            if (any) idle = 0;
            else if (++idle > 512 && cfg.handler_idle_us) usleep(cfg.handler_idle_us);
            else _mm_pause();
        }
    }
    void put_finish(int o, const PutJob &j, uint32_t ver) {
        tl_reply_stamp = cfg.diag_put_timing ? __rdtsc() : 0;
        if (j.src == o) deliver(o, j.reqid, ver, cfg.scheme == Scheme::PRIV ? priv_token(o, j.key) : 0, j.seq);
        else {
            MsgHdr m{};
            m.type = M_PUTDONE; m.nlines = 1; m.key = j.key; m.a = j.reqid; m.b = ver; m.d = j.seq; m.e = tl_reply_stamp;
            send(o, j.src, &m, nullptr, 0, false);
        }
    }
    bool lease_scheme() const {
        return cfg.rules.lease && cfg.scheme != Scheme::PRIV && cfg.scheme != Scheme::SHARD;
    }
    int exec_lanes_eff() const {
        if (!cfg.rules.put_serial) return 1;
        return std::max(1, std::min(cfg.exec_lanes, 255));
    }
    Host::Lane &lane_of(Host &X, uint64_t key) {
        return *X.lanes[X.lanes.size() == 1 ? 0 : kg_bucket(key) % X.lanes.size()];
    }
    bool put_async_on() const {
        return cfg.put_async && log_scheme() && !cfg.rules.lease && !cfg.rules.revoke && cfg.rules.put_serial &&
               be->put_latency_ns(0) != ~0ULL;
    }
    // put_async: one lane's executor as an event loop. Events, all on this thread: a backend write completes (apply it,
    // step 3), a reply is due (send it; the key's next PUT may start). New PUTs start (steps 1-2) between events.
    struct APut {
        PutJob j;
        uint64_t cut_ns = 0, cut_tsc = 0, due_tsc = 0;
        PutStamps ps{};
        uint32_t ver = 0;
        bool published = false;
        uint64_t cut_mono = 0;   // tl_cut_mono of its cut (window order: taken at the start)
    };
    void executor_loop_async(int o, int li) {
        bind(o);
        if (cfg.exec_timer_slack_ns) prctl(PR_SET_TIMERSLACK, (unsigned long)cfg.exec_timer_slack_ns, 0, 0, 0);
        Host &X = *hosts[o];
        Host::Lane &L = *X.lanes[li];
        tl_lane = li;
        const double tpn = tsc_per_ns();
        const uint64_t spin_tsc = (uint64_t)((double)cfg.put_spin_ns * tpn);
        const uint64_t W_tsc = (uint64_t)((double)log_W_ns() * tpn);
        std::multimap<uint64_t, APut *> ev;   // TSC -> PUT (backend write in flight, or published and waiting to reply)
        std::unordered_map<uint64_t, std::deque<PutJob>> busy;   // keys with a PUT in flight -> PUTs waiting for it (M5)
        static thread_local char buf[MAX_OUT];
        std::function<void(uint64_t)> release_key;
        auto start = [&](const PutJob &j) {   // (X.executing counts a PUT from its dequeue to its reply)
            const ShmCount shm0 = tl_shm;
            tl_ps = PutStamps{j.t_send, j.t_recv, cfg.diag_put_timing ? __rdtsc() : 0, 0, 0, 0, 0};
            put_begin(o, j.key, j.op);
            shm_add(X.st->shm[SHM_PUT], shm0);
            APut *p = new APut{j, tl_cut_ns, tl_cut_tsc, 0, tl_ps, 0, false};
            p->cut_mono = tl_cut_mono;
            uint64_t lat = j.op == W_EXPIRE ? 0 : be->put_latency_ns(j.key);   // EXPIRE: no backend write
            ev.emplace(tsc_now() + (uint64_t)((double)lat * tpn), p);
        };
        release_key = [&](uint64_t key) {   // M5: the key's next PUT (if any) starts now
            auto it = busy.find(key);
            if (it == busy.end()) return;
            if (it->second.empty()) { busy.erase(it); return; }
            PutJob nj = it->second.front();
            it->second.pop_front();
            start(nj);
        };
        for (;;) {
            if (stop.load(std::memory_order_relaxed)) {
                for (auto &e : ev) delete e.second;
                return;
            }
            bool did = false;
            while (!ev.empty() && ev.begin()->first <= tsc_now()) {
                APut *p = ev.begin()->second;
                ev.erase(ev.begin());
                did = true;
                if (!p->published) {   // the backend write has completed: apply it, step 3 on this (the bucket's) lane
                    const ShmCount shm0 = tl_shm;
                    tl_ps = p->ps; tl_cut_ns = p->cut_ns; tl_cut_tsc = p->cut_tsc; tl_cut_mono = p->cut_mono;
                    uint32_t vlen;
                    uint32_t ver = be_write(p->j.key, buf, &vlen, p->j.op, true);
                    put_end(o, p->j.key, ver, buf, vlen, p->j.op);
                    shm_add(X.st->shm[SHM_PUT], shm0);
                    if (cfg.diag_put_timing) tl_ps.publish = __rdtsc();
                    p->ps = tl_ps; p->cut_ns = tl_cut_ns; p->cut_tsc = tl_cut_tsc; p->ver = ver; p->published = true;
                    p->due_tsc = put_wait_on() ? tl_cut_tsc + W_tsc : 0;   // W from the cut (M4)
                    ev.emplace(std::max(p->due_tsc, tsc_now()), p);
                    if (cfg.put_key_after_publish) release_key(p->j.key);
                    continue;
                }
                // reply: never before the due nor before the publish (P2)
                if (!p->published || (p->due_tsc && tsc_now() < p->due_tsc)) {
                    X.st->reply_early++;
                    while (p->due_tsc && tsc_now() < p->due_tsc) _mm_pause();
                }
                put_finish(o, p->j, p->ver);
                if (cfg.diag_put_timing) diag_put_record(*X.st, p->ps, p->due_tsc, tl_reply_stamp);
                X.st->put_async_puts++;
                X.executing.fetch_sub(1);
                if (!cfg.put_key_after_publish) release_key(p->j.key);   // M5: only after the reply
                delete p;
            }
            if (L.njobs.load(std::memory_order_acquire) && mono_ns() >= X.writable_at.load()) {
                PutJob j;
                bool got = false;
                {
                    std::lock_guard<std::mutex> g(L.ex_mu);
                    if (!L.jobs.empty()) {
                        j = L.jobs.front();
                        L.jobs.pop_front();
                        L.njobs.fetch_sub(1, std::memory_order_relaxed);
                        got = true;
                    }
                }
                if (got) {
                    X.executing.fetch_add(1);
                    auto it = busy.find(j.key);
                    if (it != busy.end()) it->second.push_back(j);   // same key in flight: wait for its reply
                    else {
                        busy.emplace(j.key, std::deque<PutJob>());
                        start(j);
                    }
                    continue;
                }
            }
            if (did) continue;
            uint64_t now = tsc_now();
            if (L.njobs.load(std::memory_order_acquire)) {   // jobs, but the owner is not writable yet (restart_wait)
                if (ev.empty()) std::this_thread::sleep_for(std::chrono::microseconds(50));
                else _mm_pause();
                continue;
            }
            if (ev.empty() && cfg.put_idle_ns) {   // idle: spin or poll instead of waiting to be woken
                if (cfg.put_idle_ns > 1) std::this_thread::sleep_for(std::chrono::nanoseconds(cfg.put_idle_ns));
                else _mm_pause();
                continue;
            }
            if (ev.empty() || ev.begin()->first > now + spin_tsc) {
                std::unique_lock<std::mutex> g(L.ex_mu);
                if (!L.jobs.empty() || stop.load()) continue;
                if (ev.empty()) L.ex_cv.wait(g);
                else L.ex_cv.wait_for(g, std::chrono::nanoseconds((uint64_t)((double)(ev.begin()->first - now - spin_tsc) / tpn)));
                continue;
            }
            const uint64_t next = ev.begin()->first;
            while (tsc_now() < next && !L.njobs.load(std::memory_order_acquire) && !stop.load(std::memory_order_relaxed)) _mm_pause();
        }
    }
    void executor_loop(int o, int li) {
        if (put_async_on()) return executor_loop_async(o, li);
        bind(o);
        if (cfg.exec_timer_slack_ns) prctl(PR_SET_TIMERSLACK, (unsigned long)cfg.exec_timer_slack_ns, 0, 0, 0);
        Host &X = *hosts[o];
        Host::Lane &L = *X.lanes[li];
        tl_lane = li;
        using clk = std::chrono::steady_clock;
        for (;;) {
            PutJob j;
            {
                std::unique_lock<std::mutex> g(L.ex_mu);
                for (;;) {
                    while (!L.done_q.empty() && L.done_q.front().due <= clk::now()) {   // PUTs whose T has passed
                        PutDone d = L.done_q.front();
                        L.done_q.pop_front();
                        g.unlock();
                        while (d.due_tsc && tsc_now() < d.due_tsc) _mm_pause();   // log: W on the hit check's clock
                        put_finish(o, d.j, d.ver);
                        if (cfg.diag_put_timing) diag_put_record(*X.st, d.ps, d.due_tsc, tl_reply_stamp);
                        g.lock();
                    }
                    if (stop.load()) return;
                    if (!L.jobs.empty()) break;
                    if (L.done_q.empty()) L.ex_cv.wait(g);
                    else L.ex_cv.wait_until(g, L.done_q.front().due);
                }
                j = L.jobs.front();
                L.jobs.pop_front();
                L.njobs.fetch_sub(1, std::memory_order_relaxed);
            }
            X.executing.fetch_add(1);
            while (mono_ns() < X.writable_at.load() && !stop.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
            const ShmCount shm0 = tl_shm;
            tl_ps = PutStamps{j.t_send, j.t_recv, cfg.diag_put_timing ? __rdtsc() : 0, 0, 0, 0, 0};
            uint32_t ver = cfg.scheme == Scheme::PRIV    ? do_put_priv(o, j.key, j.op)
                           : cfg.scheme == Scheme::SHARD ? do_put_shard(o, j.key, j.op)
                                                         : do_put(o, j.key, j.op);
            shm_add(X.st->shm[SHM_PUT], shm0);
            if (cfg.diag_put_timing) tl_ps.publish = __rdtsc();
            if (lease_scheme() && cfg.rules.put_wait) {   // return only T after the old entry was removed / replaced
                uint64_t now = mono_ns(), from = cfg.rules.put_wait_from_cut ? tl_cut_ns : now;
                uint64_t left = from + cfg.lease_ns > now ? from + cfg.lease_ns - now : 0;
                std::lock_guard<std::mutex> g(L.ex_mu);
                L.done_q.push_back({clk::now() + std::chrono::nanoseconds(left), j, ver, 0, tl_ps});
            } else if (log_scheme() && put_wait_on()) {   // return W after the cut (overlaps the backend write)
                uint64_t due = tl_cut_tsc + (uint64_t)((double)log_W_ns() * tsc_per_ns()), now = tsc_now();
                uint64_t left = due > now ? (uint64_t)((double)(due - now) / tsc_per_ns()) : 0;
                std::lock_guard<std::mutex> g(L.ex_mu);
                PutDone d{clk::now() + std::chrono::nanoseconds(left), j, ver, due, tl_ps};
                // in due order (with one executor dues only grow; with put_serial off, a later PUT may be due earlier)
                L.done_q.insert(std::upper_bound(L.done_q.begin(), L.done_q.end(), d,
                                                 [](const PutDone &a, const PutDone &b) { return a.due < b.due; }), d);
            } else {
                put_finish(o, j, ver);
                if (cfg.diag_put_timing) diag_put_record(*X.st, tl_ps, 0, tl_reply_stamp);
            }
            X.executing.fetch_sub(1);
        }
    }
    // lease: this host's epoch, advanced every T/2 (stops while epoch_paused: test of lease_tsc)
    // Epoch e covers TSC [base + e*P, base + (e+1)*P) (P = tick period). The ticker, and with lease_clock_hit any
    // reader, raises the epoch to the current period; last_tick_tsc is the scheduled start of that period (not the
    // time the ticker happened to wake up), so a late wake-up never stretches a lease.
    double period_cycles() const { return (double)tick_ns() * tsc_per_ns(); }
    uint32_t raise_epoch(Host &X, uint64_t now) {
        double per = period_cycles();
        uint32_t target = (uint32_t)((double)(now - X.base_tsc) / per) + 1;   // epochs start at 1
        uint32_t e = X.epoch.load(std::memory_order_acquire);
        while (e < target && !X.epoch.compare_exchange_weak(e, target, std::memory_order_acq_rel)) {}
        if (e < target) e = target;
        uint64_t sched = X.base_tsc + (uint64_t)((double)(e - 1) * per);
        uint64_t last = X.last_tick_tsc.load(std::memory_order_acquire);
        while (last < sched && !X.last_tick_tsc.compare_exchange_weak(last, sched, std::memory_order_acq_rel)) {}
        return e;
    }
    void ticker_loop(int h) {
        Host &X = *hosts[h];
        timespec next;
        clock_gettime(CLOCK_MONOTONIC, &next);
        const long half = (long)tick_ns();
        while (!stop.load(std::memory_order_relaxed)) {
            next.tv_nsec += half;
            while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
            for (;;) {   // sleep until `next` in slices of <= 10 ms, so stop() never waits for a long period
                uint64_t want = (uint64_t)next.tv_sec * 1000000000ULL + next.tv_nsec, now = mono_ns();
                if (now >= want || stop.load(std::memory_order_relaxed)) break;
                uint64_t until = std::min<uint64_t>(want, now + 10'000'000);
                timespec u = {(time_t)(until / 1000000000ULL), (long)(until % 1000000000ULL)};
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &u, nullptr);
            }
            if (stop.load(std::memory_order_relaxed)) break;
            {
                uint64_t want = (uint64_t)next.tv_sec * 1000000000ULL + next.tv_nsec, now = mono_ns();
                if (now > want + (uint64_t)half / 10) {
                    X.st->tick_late++;
                    uint64_t late = now - want, m = X.st->tick_late_max_ns.load();
                    while (late > m && !X.st->tick_late_max_ns.compare_exchange_weak(m, late)) {}
                }
            }
            if (X.epoch_paused.load()) continue;
            raise_epoch(X, __rdtsc());
        }
    }
    uint64_t tick_ns() const { return std::max<uint64_t>(cfg.lease_tick_ns ? cfg.lease_tick_ns : cfg.lease_ns / 2, 1000); }
    // lease_clock_hit: the current epoch, advanced by this reader if the ticker is overdue
    uint32_t epoch_now(Host &X) {
        uint32_t e = X.epoch.load(std::memory_order_acquire);
        if (!cfg.rules.lease_clock_hit) return e;
        uint64_t now = __rdtsc();
        uint64_t last = X.last_tick_tsc.load(std::memory_order_acquire);
        if (now > last && (double)(now - last) >= period_cycles()) e = raise_epoch(X, now);
        return e;
    }
    // lease_tsc: called at the start of a slow path. If the epoch thread is behind by more than T (it did not
    // tick for T although it should every T/2), raise the epoch to the clock ourselves: every entry older than T
    // expires.
    void epoch_check(Host &X) {
        uint64_t now = __rdtsc(), last = X.last_tick_tsc.load(std::memory_order_acquire);
        if (now > last && (double)(now - last) > (double)cfg.lease_ns * tsc_per_ns()) {
            // the ticker is (or was) stuck for > T = 2 periods: catch the epoch up with the clock (>= 2 epochs), so
            // every entry older than T expires
            raise_epoch(X, now);
            X.st->epoch_stall++;
        }
    }
    // ---------------------------------------------------------------- log: reader host's polling thread
    static size_t rec_bucket(uint64_t key) { return mix64(key ^ 0x7f4a7c159e3779b9ULL) % REC_BUCKETS; }
    // invalidate every local entry of owner o at once: entries carry the drop generation they were installed under
    void log_drop_all(Host &X, int o) {
        if (X.li.c16 && ((X.lv[o].drop_gen.load(std::memory_order_relaxed) + 1) & X.li.ep_mask) == 0) li16_clear_owner(X, o);
        X.lv[o].drop_gen.fetch_add(1, std::memory_order_acq_rel);
        X.lv[o].exam_lb = X.lv[o].count;
        X.st->log_dropall++;
    }
    // apply one record (poll_mu held): remember it for in-flight slow paths, delete the local entry
    void log_apply(Host &X, int o, uint64_t key) {
        LogView &L = X.lv[o];
        uint64_t a = L.applied.load(std::memory_order_relaxed) + 1;
        L.rec_seq[rec_bucket(key)] = a;
        L.applied.store(a, std::memory_order_release);
        X.st->log_applied++;
        if (!X.li.bloom_maybe(key, o)) { X.st->log_bloom_skip++; return; }   // never installed: nothing to delete
        Stripe &s = stripe(X.id, key);
        std::lock_guard<std::mutex> g(s.mu);
        if (X.li.mark_deleted(key, o)) X.st->log_deleted++;
    }
    // refresh record (poll_mu held): counts as a record for the key (install check); repoints the entry only if this
    // host holds the key (valid or marked deleted, not dropped): flush the object's lines first, then repoint
    void log_refresh(Host &X, int o, const LogRec &r) {
        LogView &L = X.lv[o];
        uint64_t a = L.applied.load(std::memory_order_relaxed) + 1;
        L.rec_seq[rec_bucket(r.key)] = a;
        L.applied.store(a, std::memory_order_release);
        X.st->log_applied++;
        if (!X.li.bloom_maybe(r.key, o)) { X.st->log_bloom_skip++; return; }   // not held: nothing to repoint
        Stripe &s = stripe(X.id, r.key);
        std::lock_guard<std::mutex> g(s.mu);
        uint64_t v;
        uint32_t ep;
        LiRef lr;
        if (!X.li.find(r.key, o, &v, &ep, &lr) || ep != X.li.ep_of(L.drop_gen.load(std::memory_order_relaxed))) return;
        if (cfg.log_refresh_hot_only && !lr.ref()) {   // log_refresh_hot_only (default off): unread entry -> delete, no flush
            if (X.li.mark_deleted(r.key, o)) X.st->log_refresh_cold++;
            return;
        }
        uint64_t c0 = __rdtsc();
        uint64_t off = (uint64_t)(r.w0 & OFF_MASK) * LINE;
        uint32_t cls = r.w0 >> 28;
        if (cls >= (uint32_t)NCLS) return;
        if (cfg.rules.refresh_flush) obj_flush(X.id, o, off, CLS[cls]);
        X.li.put(r.key, o, li_pack(off, r.gen, (int)cls), ep);
        X.st->log_refreshed++;
        X.st->refresh_cycles += __rdtsc() - c0;
    }
    // local-index cap (poll_mu held): CLOCK over the keys this host holds (queue in install order) until at most cap entries are
    // held; a read sets an entry's bit, the hand clears it (second chance) or, if clear, drops the entry (its next read
    // takes the slow path). Keys no longer held (swept, erased after a failed check) leave the queue when the hand meets them.
    void li_cap(Host &X) {
        uint64_t objs = 0;
        for (int o = 0; o < owner_n(); o++) objs += X.lv[o].objs;
        uint64_t cap = cfg.li_cap_factor > 0 ? std::max<uint64_t>(cfg.li_cap_min, (uint64_t)(cfg.li_cap_factor * objs)) : 0;
        int64_t cnt = X.li.count();
        X.st->li_cap = cap;
        X.st->li_count = cnt > 0 ? cnt : 0;
        if (!cap) {
            if (!X.clock_qi.empty()) std::fill(X.clock_inq_i.begin(), X.clock_inq_i.end(), 0);
            X.clock_q.clear(); X.clock_qi.clear(); X.clock_inq_k.clear();
            return;
        }
        if (X.li.c16) { li16_cap(X, cnt, cap); return; }
        LiTable &T = X.li.a;
        if (cnt > (int64_t)cap) {
            uint64_t excess = cnt - cap;
            for (size_t j = 0, n = X.clock_q.size(); j < n && excess && !X.clock_q.empty(); j++) {
                uint64_t k = X.clock_q.front();
                X.clock_q.pop_front();
                Stripe &st = stripe(X.id, k);
                std::lock_guard<std::mutex> g(st.mu);
                uint64_t v;
                LiSlot *sl = nullptr;
                if (!T.find(k, &v, nullptr, &sl)) { if (cfg.li_clock_inq) X.clock_inq_k.erase(k); continue; }   // not held any more
                if (sl->ref.load(std::memory_order_relaxed)) {
                    sl->ref.store(0, std::memory_order_relaxed);
                    X.clock_q.push_back(k);
                } else if (T.erase(k, v)) {
                    X.st->li_evicted++;
                    excess--;
                    if (cfg.li_clock_inq) X.clock_inq_k.erase(k);
                } else if (cfg.li_clock_inq) {
                    X.clock_inq_k.erase(k);   // not pushed back: the key left the queue
                }
            }
        }
        cnt = X.li.count();
        X.st->li_count = cnt > 0 ? cnt : 0;
        X.st->li_clock_q = X.clock_q.size();
        if (!cfg.li_clock_inq && X.clock_q.size() > 2 * (uint64_t)std::max<int64_t>(cnt, 0) + 4096) {   // drop stale and duplicate keys
            X.st->li_clock_dedups++;
            std::unordered_set<uint64_t> seen;
            std::deque<uint64_t> q;
            for (uint64_t k : X.clock_q) {
                uint64_t v;
                if (seen.insert(k).second && T.find(k, &v)) q.push_back(k);
            }
            X.clock_q.swap(q);
        }
    }
    // li16: the same CLOCK over slot indices (an index whose slot was freed and reused by another key just stands for
    // that entry now). Entries are dropped slot-wise (checked against the value read, under the slot's seqlock).
    void li16_cap(Host &X, int64_t cnt, uint64_t cap) {
        LiTable16 &T = X.li.b;
        if (cnt > (int64_t)cap) {
            uint64_t excess = cnt - cap;
            for (size_t j = 0, n = X.clock_qi.size(); j < n && excess && !X.clock_qi.empty(); j++) {
                uint32_t i = X.clock_qi.front();
                X.clock_qi.pop_front();
                uint64_t m, v;
                LiTable16::read(T.s[i], &m, &v);
                if (l16::fp(m) <= l16::FP_DEL) { if (cfg.li_clock_inq) X.clock_inq_i[i] = 0; continue; }   // nothing held there any more
                if (m & l16::REF) {
                    T.s[i].m.fetch_and(~l16::REF, std::memory_order_relaxed);
                    X.clock_qi.push_back(i);
                } else if (T.apply_at(i, m & l16::TAG_MASK, true, v, 0)) {
                    X.st->li_evicted++;
                    excess--;
                    if (cfg.li_clock_inq) X.clock_inq_i[i] = 0;
                } else {
                    X.clock_qi.push_back(i);   // changed under us (a fast-path failure marked it): next round
                }
            }
        }
        cnt = X.li.count();
        X.st->li_count = cnt > 0 ? cnt : 0;
        X.st->li_clock_q = X.clock_qi.size();
        if (!cfg.li_clock_inq && X.clock_qi.size() > 2 * (uint64_t)std::max<int64_t>(cnt, 0) + 4096) {   // drop empty and duplicate slots
            X.st->li_clock_dedups++;
            std::unordered_set<uint32_t> seen;
            std::deque<uint32_t> q;
            for (uint32_t i : X.clock_qi) {
                uint64_t m, v;
                LiTable16::read(T.s[i], &m, &v);
                if (l16::fp(m) > l16::FP_DEL && seen.insert(i).second) q.push_back(i);
            }
            X.clock_qi.swap(q);
        }
    }
    // li16: drop every entry of owner o (poll_mu held), before o's drop generation reaches a multiple of 2^16, so that
    // no entry's 16-bit ep can match again after a full wrap
    void li16_clear_owner(Host &X, int o) {
        LiTable16 &T = X.li.b;
        for (uint64_t i = 0; i <= T.mask; i++) {
            uint64_t m, v;
            LiTable16::read(T.s[i], &m, &v);
            if (l16::fp(m) > l16::FP_DEL && l16::owner(m) == o) T.apply_at(i, m & l16::TAG_MASK, false, 0, 0);
        }
        X.st->li16_owner_clears++;
    }
    // one owner's part of a poll round (poll_mu held)
    // pre: o's tail line items as read by poll_round after its batched flush (log_batch_tails); else flush + read here
    void poll_owner(Host &X, int o, const TailEnt *pre = nullptr) {
        LogView &L = X.lv[o];
        TailEnt t, t2;
        if (pre) {
            t = pre[0];
            t2 = pre[1];
        } else {
            char *tl = owner_base(o) + off_log;
            flush_line(tl);
            _mm_mfence();
            load16v(tl, &t);
            load16v(tl + 16, &t2);   // same line, already flushed: the owner's object count
            tl_shm.rd++;
        }
        L.objs = t2.te;
        uint64_t tail = t.te & TAIL_MASK;
        uint32_t ep = (uint32_t)(t.te >> 40);
        if (!L.seen) {   // first poll: this host has no entries of o yet, older records do not matter
            L.seen = true;
            L.epoch = ep;
            L.pos = tail;
            L.count = L.exam_lb = L.pass_start = t.count;
            return;
        }
        L.count = t.count;
        if (ep != L.epoch && cfg.rules.log_epoch) {   // new owner / restart: a new log and a new counter
            L.epoch = ep;
            L.pos = tail;
            log_drop_all(X, o);
            L.pass_start = t.count;
            X.st->log_epoch_chg++;
            return;
        }
        const uint64_t U = cfg.log_rec8 ? SPL : RPL;   // tail units (slots / records) per line
        const uint64_t cap = (uint64_t)cfg.log_lines * U;
        if (tail > L.pos) {
            if (cfg.rules.log_overrun && tail - L.pos > cap) {   // lapped: records were overwritten unread
                log_drop_all(X, o);
                X.st->log_overrun++;
            } else {
                // as for objects, every record line is flushed before its one 64 B load; up to log_prefetch_lines lines
                // at a time (all flushes, one mfence, then the loads) so their misses overlap
                const uint64_t K = std::max<uint64_t>(1, std::min<uint64_t>(cfg.log_prefetch_lines, 64));
                const uint64_t last = (tail - 1) / U;
                alignas(64) char buf[64][LINE];
                uint64_t bl = 0, bn = 0;   // buffered record lines [bl, bl + bn)
                const char *line = nullptr;
                uint64_t cur = ~0ULL;
                uint64_t hdr = 0;
                for (uint64_t i = L.pos; i < tail; i++) {
                    uint64_t li = i / U;
                    if (li != cur) {
                        if (li < bl || li >= bl + bn) {
                            uint64_t n = std::min<uint64_t>(K, last - li + 1);
                            for (uint64_t j = 0; j < n; j++) flush_line(log_line(o, (li + j) % cfg.log_lines));
                            _mm_mfence();
                            for (uint64_t j = 0; j < n; j++) load64(log_line(o, (li + j) % cfg.log_lines), buf[j]);
                            tl_shm.rd += n;
                            bl = li;
                            bn = n;
                        }
                        line = buf[li - bl];
                        cur = li;
                        memcpy(&hdr, line, 8);
                        uint64_t lap = cfg.log_rec8 ? hdr & LAP_MASK : hdr;
                        if (cfg.rules.log_overrun && lap != li / cfg.log_lines + 1) {   // overwritten by a later lap
                            log_drop_all(X, o);
                            X.st->log_overrun++;
                            break;
                        }
                    }
                    LogRec r;
                    if (cfg.log_rec8) {
                        const uint64_t pos = i % SPL;
                        uint64_t w;
                        memcpy(&w, line + 8 + pos * 8, 8);
                        if (hdr >> (56 + pos) & 1) {   // refresh record: key, then (w0 << 32 | generation)
                            uint64_t b;
                            memcpy(&b, line + 8 + (pos + 1) * 8, 8);
                            r = LogRec{w, (uint32_t)(b >> 32), (uint32_t)b};
                            i++;
                        } else if (w) {
                            r = LogRec{w, DEL_W0, 0};
                        } else {
                            continue;   // pad slot
                        }
                    } else {
                        memcpy(&r, line + 16 + (i % RPL) * 16, sizeof r);
                    }
                    if (r.w0 == DEL_W0) log_apply(X, o, r.key);
                    else log_refresh(X, o, r);
                }
            }
            L.pos = tail;
        }
        // (tail < pos happens only when an epoch change went unnoticed: log_epoch off)
        // R2: an entry is examined at least once per sweep pass; if the counter may have moved half a round since an
        // entry's last examination, its age mod 2^gen_bits says nothing: drop them all
        if (cfg.rules.log_dropall_lag && t.count - L.exam_lb >= gen_round() / 2) {
            log_drop_all(X, o);
            X.st->log_lag_dropall++;
        }
    }
    // R2 sweep of the next chunk of the local index (poll_mu held): drop entries invalidated by a drop-all, and entries
    // whose generation is >= half a round old against their owner's counter read in this round
    // slots per poll: log_sweep_slots, scaled down for intervals below log_sweep_ref_ns (same slots per second)
    uint64_t sweep_slots() const { return sweep_slots_of(cfg); }
    static uint64_t interval_of(const Config &c) {
        return std::max<uint64_t>(c.log_interval_ns ? c.log_interval_ns : c.log_poll_ns / 2, 1000);
    }
    static uint64_t sweep_slots_of(const Config &c) {
        uint64_t s = c.log_sweep_slots, iv = interval_of(c);
        if (s && c.log_sweep_ref_ns && iv < c.log_sweep_ref_ns)
            s = std::max<uint64_t>(16, (uint64_t)((double)s * (double)iv / (double)c.log_sweep_ref_ns));
        return s;
    }
    // seconds per pass over the local index, and the margin of the R2 condition (rackcache.h log_sweep_slots)
    static double sweep_pass_s(const Config &c) {
        uint64_t s = sweep_slots_of(c), n = c.li_slots;
        uint64_t polls = s ? (n + s - 1) / s : 1;
        return (double)polls * (double)interval_of(c) / 1e9;
    }
    static double sweep_margin_of(const Config &c) {
        double adv = c.log_max_alloc_per_s * sweep_pass_s(c);
        return adv > 0 ? std::ldexp(1.0, c.gen_bits - 1) / adv : 1e300;
    }
    void sweep_chunk(Host &X) {
        const uint64_t n = (X.li.c16 ? X.li.b.mask : X.li.a.mask) + 1, slots = sweep_slots();
        const uint64_t chunk = slots ? std::min<uint64_t>(slots, n) : n;
        const uint32_t m = gen_mask();
        const uint64_t half = gen_round() / 2;
        if (X.li_sweep_pos == 0)
            for (int o = 0; o < H; o++) X.lv[o].pass_start = X.lv[o].count;
        for (uint64_t j = 0; j < chunk; j++) {
            uint64_t i = X.li_sweep_pos++;
            if (X.li.c16) {   // li16: owner and ep from the slot; dropped slot-wise (checked against the value read)
                LiTable16 &T = X.li.b;
                uint64_t w, v;
                LiTable16::read(T.s[i], &w, &v);
                if (l16::fp(w) > l16::FP_DEL) {
                    int o = l16::owner(w);
                    LogView &L = X.lv[o];
                    bool drop = l16::ep(w) != X.li.ep_of(L.drop_gen.load(std::memory_order_relaxed));
                    bool old = !drop && !xtra_own() && (uint32_t)v && cfg.rules.log_sweep && (((uint32_t)L.count - (uint32_t)v) & m) >= half;
                    if ((drop || old) && T.apply_at(i, w & l16::TAG_MASK, true, v, 0) && old) X.st->sweep_dropped++;
                }
            } else {
                uint64_t k, v;
                uint32_t ep;
                LiTable::read(X.li.a.s[i], &k, &v, &ep);
                if (k != LI_EMPTY && k != LI_DEL) {
                    int o = Cache::owner_of_static(k, owner_n());
                    LogView &L = X.lv[o];
                    bool drop = ep != L.drop_gen.load(std::memory_order_relaxed);
                    bool old = !drop && !xtra_own() && (uint32_t)v && cfg.rules.log_sweep && (((uint32_t)L.count - (uint32_t)v) & m) >= half;
                    if (drop || old) {
                        Stripe &s = stripe(X.id, k);
                        std::lock_guard<std::mutex> g(s.mu);
                        if (X.li.a.erase(k, v) && old) X.st->sweep_dropped++;
                    }
                }
            }
            if (X.li_sweep_pos == n) {   // pass complete: every entry was examined at counter >= pass_start
                X.li_sweep_pos = 0;
                for (int o = 0; o < H; o++) X.lv[o].exam_lb = std::max(X.lv[o].exam_lb, X.lv[o].pass_start);
                break;
            }
        }
    }
    // xtra_own: the poller's housekeeping only -- tail lines for the allocation count and object count (R2 lag check,
    // local-index cap), the R2 sweep (scaled to the longer interval), the cap; no records, no heartbeat
    void hk_round(int h) {
        Host &X = *hosts[h];
        std::lock_guard<std::mutex> g(X.poll_mu);
        const ShmCount shm0 = tl_shm;
        const uint64_t c0 = __rdtsc();
        for (int o = 0; o < H; o++) {
            char *tl = owner_base(o) + off_log;
            flush_line(tl);
            _mm_mfence();
            TailEnt t, t2;
            load16v(tl, &t);
            load16v(tl + 16, &t2);
            tl_shm.rd++;
            LogView &L = X.lv[o];
            L.objs = t2.te;
            const uint32_t ep = (uint32_t)(t.te >> 40);
            if (!L.seen) {
                L.seen = true;
                L.epoch = ep;
                L.count = L.exam_lb = L.pass_start = t.count;
                continue;
            }
            L.count = t.count;
            if (ep != L.epoch && cfg.rules.log_epoch) {   // owner restart: a new counter
                L.epoch = ep;
                log_drop_all(X, o);
                L.pass_start = t.count;
                continue;
            }
            if (cfg.rules.log_dropall_lag && t.count - L.exam_lb >= gen_round() / 2) {
                log_drop_all(X, o);
                X.st->log_lag_dropall++;
            }
        }
        const uint64_t k = std::max<uint64_t>(1, cfg.xtra_hk_ns / log_interval());   // same sweep rate as the poller's
        for (uint64_t j = 0; j < k; j++) sweep_chunk(X);
        li_cap(X);
        X.poll_seq.fetch_add(1, std::memory_order_relaxed);
        X.st->polls++;
        shm_add(X.st->shm[SHM_POLL], shm0);
        X.st->poll_cycles += __rdtsc() - c0;
    }
    void poll_round(int h) {
        if (!log_scheme()) return;   // no log, no poller
        if (xtra_own()) return hk_round(h);
        Host &X = *hosts[h];
        std::lock_guard<std::mutex> g(X.poll_mu);
        const ShmCount shm0 = tl_shm;
        uint64_t t0 = tsc_now();   // the heartbeat: taken BEFORE any tail is read (before the first flush)
        const uint64_t c0 = t0;
        if (cfg.log_batch_tails) {   // flush all tail lines, one mfence, then read them (any order is fine)
            TailEnt *tt = X.tail_buf.data();
            for (int o = 0; o < H; o++) flush_line(owner_base(o) + off_log);
            _mm_mfence();
            for (int o = 0; o < H; o++) {
                const char *tl = owner_base(o) + off_log;
                load16v(tl, &tt[2 * o]);
                load16v(tl + 16, &tt[2 * o + 1]);
                tl_shm.rd++;
            }
            hook(Hook::PollAfterTails, h, 0);
            if (!cfg.rules.log_ts_before_tails) t0 = tsc_now();   // ablation (M4): timestamp after the tail reads (test M4-tsbefore)
            for (int o = 0; o < H; o++) poll_owner(X, o, &tt[2 * o]);
        } else {
            for (int o = 0; o < H; o++) {
                poll_owner(X, o);
                if (o == 0) {
                    hook(Hook::PollAfterTails, h, 0);
                    if (!cfg.rules.log_ts_before_tails) t0 = tsc_now();   // ablation (M4): after the first tail
                }
            }
        }
        sweep_chunk(X);
        li_cap(X);
        if (X.li.bloom && X.li.bloom_adds > cfg.li_slots) {   // log_bloom_bits: rebuild between two rounds, under poll_mu
            X.li.bloom_rebuild();
            X.st->bloom_rebuilds++;
        }
        X.hb.store(t0, std::memory_order_release);
        X.poll_seq.fetch_add(1, std::memory_order_relaxed);
        X.st->polls++;
        shm_add(X.st->shm[SHM_POLL], shm0);
        X.st->poll_cycles += __rdtsc() - c0;
        if (cfg.diag_lock_timing) diag_add(*X.st, 2, __rdtsc() - c0);   // poll_mu held from before c0 to the return
    }
    // this host's poller thread CPU time so far (CLOCK_THREAD_CPUTIME_ID: rounds, sleeps and wake-ups), into Stats
    static void poll_cpu(Host &X) {
        timespec t;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
        X.st->poll_cpu_ns.store((uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec, std::memory_order_relaxed);
    }
    void poller_loop(int h) {
        Host &X = *hosts[h];
        uint64_t cpu_n = 0;
        const uint64_t iv = xtra_own() ? std::max<uint64_t>(cfg.xtra_hk_ns, 1000) : log_interval();
        // Linux's default timer slack (50 us) would make short sleeps overshoot by up to that much: 1 ns for short intervals
        if (iv < 100'000) prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
        uint64_t next = mono_ns();
        while (!stop.load(std::memory_order_relaxed)) {
            next += iv;
            for (;;) {   // sleep until `next` in slices of <= 10 ms
                uint64_t now = mono_ns();
                if (now >= next || stop.load(std::memory_order_relaxed)) break;
                uint64_t until = std::min<uint64_t>(next, now + 10'000'000);
                timespec u = {(time_t)(until / 1000000000ULL), (long)(until % 1000000000ULL)};
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &u, nullptr);
            }
            if (stop.load(std::memory_order_relaxed)) break;
            uint64_t now = mono_ns();
            if (now > next + iv / 10) {
                X.st->tick_late++;
                uint64_t late = now - next, m = X.st->tick_late_max_ns.load();
                while (late > m && !X.st->tick_late_max_ns.compare_exchange_weak(m, late)) {}
                if (now > next + iv) next = now;   // do not burst to catch up
            }
            if (X.poll_paused.load()) continue;
            poll_round(h);
            if ((++cpu_n & 15) == 0) poll_cpu(X);
        }
        poll_cpu(X);
    }
    // the current heat window of host X
    uint16_t heat_window(Host &X) {
        if (log_scheme()) return (uint16_t)(X.poll_seq.load(std::memory_order_relaxed) / std::max<uint32_t>(1, cfg.heat_window_polls));
        return (uint16_t)((double)__rdtsc() / ((double)cfg.heat_window_ns * tsc_per_ns()));
    }
    // hit check: the heartbeat h0 was read before the lookup; after the loads, now - h <= P
    bool hb_ok(Host &X, uint64_t h0) {
        uint64_t h = cfg.rules.log_hb_before ? h0 : X.hb.load(std::memory_order_acquire);
        if (cfg.rules.log_seqlock && X.hb.load(std::memory_order_acquire) != h) return false;
        if (!cfg.rules.log_hit_clock) return true;
        uint64_t now = cfg.hit_clock == 1 ? tsc_after_loads() : tsc_now();
        return (int64_t)(now - h) <= (int64_t)log_P_cycles();
    }

    int new_mailbox(int h) {
        Host &X = *hosts[h];
        std::lock_guard<std::mutex> g(X.mb_mu);
        X.mbox.emplace_back(new Mailbox);
        return (int)X.mbox.size() - 1;
    }
    Mailbox *mailbox(int h, int id) {
        Host &X = *hosts[h];
        std::lock_guard<std::mutex> g(X.mb_mu);
        return X.mbox[id].get();
    }
    uint32_t wait_mailbox(Mailbox *mb, uint64_t timeout_ns = 0) {
        int spins = 0;
        uint64_t t0 = timeout_ns ? mono_ns() : 0;
        while (!mb->ready.load(std::memory_order_acquire)) {
            if (stop.load(std::memory_order_relaxed)) return 0;
            if (timeout_ns && (spins & 255) == 0 && mono_ns() - t0 > timeout_ns) return 0;
            if (++spins > 4096) { std::this_thread::yield(); spins = 0; } else _mm_pause();
        }
        return mb->val;
    }
};

// ---------------------------------------------------------------- Cache
Cache::Cache(const Config &cfg, Backend *be) : cfg_(cfg), p_(new Impl(cfg, be)) {}
Cache::~Cache() { stop(); }
// key -> owner: MurmurHash3 fmix64 of the key, the same owner hash as our hit-ratio simulators, so that the
// replayer's hit rates are directly comparable with theirs.
// Optional salt of the key -> owner hash, RC_OWNER_SALT in the environment
// (replay --owner-salt sets it before any cache exists, so every host process sees it); default 0 = the hash is unchanged.
// Lets a run move a dominant key (one that takes a large share of the GETs) off the measured host's owner.
static uint64_t owner_salt() {
    static const uint64_t s = [] { const char *e = getenv("RC_OWNER_SALT"); return e ? (uint64_t)strtoull(e, nullptr, 0) : 0ULL; }();
    return s;
}
int Cache::owner_of_static(uint64_t key, int hosts) {
    uint64_t h = key ^ owner_salt();
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL; h ^= h >> 33; h *= 0xC4CEB53A85E1FE53ULL; h ^= h >> 33;
    return (int)(h % (uint64_t)hosts);
}
int Cache::owner_of(uint64_t key) const { return owner_of_static(key, cfg_.owners > 0 ? cfg_.owners : cfg_.hosts); }
Stats &Cache::stats(int h) {
    if (cfg_.shared_stats) return cfg_.shared_stats[h];
    return *p_->hosts[h]->st;
}
size_t Cache::shm_create(const Config &cfg) {
    Impl::check_cfg(cfg);
    size_t bytes = Impl::layout(cfg).total;
    int fd = shm_open(cfg.shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) { perror("shm_open create"); exit(1); }
    if (ftruncate(fd, (off_t)bytes)) { perror("ftruncate"); exit(1); }
    char *p = (char *)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    Impl::bind_and_touch(p, bytes, cfg.shm_node);   // shmem keeps the policy; pages are placed now
    munmap(p, bytes);
    return bytes;
}
void Cache::shm_unlink(const std::string &name) { ::shm_unlink(name.c_str()); }
void Cache::start() {
    p_->stop.store(false);
    tsc_per_ns();   // calibrate before any thread needs it
    for (int h = 0; h < p_->H; h++) {   // kg_cxl: an owner (re)starting continues its kg words from CXL
        if (!p_->hosts[h] || !p_->kg_cxl()) continue;
        Host &X = *p_->hosts[h];
        std::lock_guard<std::mutex> g(X.mu);
        for (uint64_t b = 0; b < cfg_.kg_buckets; b++) X.kgb[b] = vld32(p_->kg_word(h, b));
    }
    for (int h = 0; h < p_->H; h++) {   // log: (re)start this owner's log in a new epoch
        if (!p_->hosts[h] || !p_->log_scheme()) continue;
        Host &X = *p_->hosts[h];
        std::lock_guard<std::mutex> g(X.mu);
        TailEnt t;
        load16v(p_->owner_base(h) + p_->off_log, &t);
        X.log_epoch = (uint32_t)(t.te >> 40) + 1;
        X.log_tail = 0;
        p_->log_publish(h);
    }
    for (auto &x : p_->hosts)   // every lane exists before any handler / client can queue a PUT
        if (x) for (int li = 0; li < p_->exec_lanes_eff(); li++) x->lanes.push_back(std::make_unique<Host::Lane>());
    for (int h = 0; h < p_->H; h++) {
        if (!p_->hosts[h]) continue;
        Host &X = *p_->hosts[h];
        uint64_t wait = p_->lease_scheme() ? cfg_.lease_ns : p_->log_scheme() ? p_->log_W_ns() : 0;
        X.writable_at = cfg_.rules.restart_wait ? mono_ns() + wait : 0;
        X.handler = std::thread([this, h] { p_->handler_loop(h); });
        for (size_t li = 0; li < X.lanes.size(); li++) X.lanes[li]->th = std::thread([this, h, li] { p_->executor_loop(h, (int)li); });
        if (!p_->cfg.rules.put_serial) X.executor2 = std::thread([this, h] { p_->executor_loop(h, 0); });
        if (p_->lease_scheme()) {
            X.base_tsc = __rdtsc();
            X.last_tick_tsc = X.base_tsc;
            X.epoch = 1;
            X.ticker = std::thread([this, h] { p_->ticker_loop(h); });
        }
        if (p_->log_scheme()) {
            p_->poll_round(h);   // first round: records the logs' positions, publishes a heartbeat
            X.poller = std::thread([this, h] { p_->poller_loop(h); });
        }
    }
}
void Cache::stop() {
    if (!p_) return;
    p_->stop.store(true);
    for (auto &x : p_->hosts) {
        if (!x) continue;
        for (auto &l : x->lanes) { std::lock_guard<std::mutex> g(l->ex_mu); l->ex_cv.notify_all(); }
        if (x->handler.joinable()) x->handler.join();
        for (auto &l : x->lanes) if (l->th.joinable()) l->th.join();
        if (x->executor2.joinable()) x->executor2.join();
        if (x->ticker.joinable()) x->ticker.join();
        if (x->poller.joinable()) x->poller.join();
    }
}
void Cache::peer_access(int o, int kind, int where) {
    const Impl &P = *p_;
    const size_t off[] = {0, P.off_ai, P.off_rings, P.off_tails, P.off_log, P.off_kg, P.per_owner - LINE};
    volatile char *p = P.owner_base(o) + off[where < 0 || where > 6 ? 0 : where];
    switch (kind) {
    case 0: (void)*p; break;
    case 1: _mm_clflush((const void *)p); _mm_mfence(); break;
    case 2: _mm_clflushopt((void *)p); _mm_mfence(); break;
    case 3: _mm_clwb((void *)p); _mm_sfence(); break;
    default: { char c = *p; *p = c; break; }
    }
}
bool Cache::owner_evict(uint64_t key) {
    int o = owner_of(key);
    Host &X = *p_->hosts[o];
    std::lock_guard<std::mutex> g(X.mu);
    auto it = X.objs.find(key);
    if (it == X.objs.end()) return false;
    p_->evict_obj(o, it->second);
    return true;
}
bool Cache::held_at(int h, uint64_t key) {
    if (p_->cfg.scheme == Scheme::PRIV || p_->cfg.scheme == Scheme::SHARD) {
        Priv &V = p_->hosts[h]->priv;
        std::lock_guard<std::mutex> g(V.mu);
        return V.objs.count(key) != 0;
    }
    return owner_cached(key);
}
bool Cache::owner_cached(uint64_t key) {
    Host &X = *p_->hosts[owner_of(key)];
    std::lock_guard<std::mutex> g(X.mu);
    return X.objs.count(key) != 0;
}
uint64_t Cache::owner_used(int o) {
    std::lock_guard<std::mutex> g(p_->hosts[o]->mu);
    return p_->hosts[o]->used;
}
size_t Cache::owner_objects(int o) {
    std::lock_guard<std::mutex> g(p_->hosts[o]->mu);
    return p_->hosts[o]->objs.size();
}
int64_t Cache::owner_offset(uint64_t key) {
    Host &X = *p_->hosts[owner_of(key)];
    std::lock_guard<std::mutex> g(X.mu);
    auto it = X.objs.find(key);
    return it == X.objs.end() ? -1 : (int64_t)it->second->off;
}
void Cache::wait_writable() {
    for (auto &x : p_->hosts)
        if (x) while (mono_ns() < x->writable_at.load()) std::this_thread::sleep_for(std::chrono::microseconds(100));
}
void Cache::pause_epoch(int h, bool paused) { p_->hosts[h]->epoch_paused = paused; }
void Cache::pause_poller(int h, bool paused) {
    Host &X = *p_->hosts[h];
    X.poll_paused = paused;
    std::lock_guard<std::mutex> g(X.poll_mu);   // a round in progress has finished
}
void Cache::poll_once(int h) { p_->poll_round(h); }
uint32_t Cache::owner_advance(int o, uint64_t n) {
    Host &X = *p_->hosts[o];
    std::lock_guard<std::mutex> g(X.mu);
    for (uint64_t i = 0; i < n; i++) p_->new_gen(o);
    return p_->peek_gen(X);
}
uint32_t Cache::owner_next_gen(int o) {
    Host &X = *p_->hosts[o];
    std::lock_guard<std::mutex> g(X.mu);
    return p_->peek_gen(X);
}
uint32_t Cache::owner_gen(uint64_t key) {
    Host &X = *p_->hosts[owner_of(key)];
    std::lock_guard<std::mutex> g(X.mu);
    auto it = X.objs.find(key);
    return it == X.objs.end() ? 0 : it->second->gen;
}
void Cache::owner_reset(int o) {
    Impl &P = *p_;
    Host &X = *P.hosts[o];
    std::lock_guard<std::mutex> g(X.mu);
    std::vector<Obj *> all;
    for (auto &kv : X.objs) all.push_back(kv.second);
    for (Obj *ob : all) P.evict_obj(o, ob);   // no records
    X.log_tail = 0;
    X.log_epoch++;   // the counter continues (a new owner reads it from the tail line), so only the epoch tells readers
    if (P.log_scheme()) P.log_publish(o);
    uint64_t wait = P.lease_scheme() ? cfg_.lease_ns : P.log_scheme() ? P.log_W_ns() : 0;
    if (cfg_.rules.restart_wait) X.writable_at = mono_ns() + wait;
}
bool Cache::ai_peek(uint64_t key, uint64_t *off, uint32_t *gen) {
    Impl::AiView a;
    if (p_->ai_lookup(owner_of(key), key, &a, false) != 1) return false;
    *off = a.off;
    *gen = a.gen;
    return true;
}
void Cache::emu_drop_line(int h, int o, uint64_t off) {
    Host &X = *p_->hosts[h];
    std::lock_guard<std::mutex> g(X.emu_mu);
    X.emu.erase((uint64_t)o << 40 | off);
}
void Cache::set_unreachable(int h, bool u) { p_->hosts[h]->unreachable = u; }
uint32_t Cache::epoch(int h) { return p_->hosts[h]->epoch.load(); }
void Cache::wait_idle() {
    Impl &P = *p_;
    for (int stable = 0; stable < 2;) {
        bool idle = true;
        for (int r = 0; r < P.H && idle; r++) {
            Host &X = *P.hosts[r];
            if (X.overflow_n.load() || X.executing.load()) { idle = false; break; }
            for (auto &l : X.lanes) { std::lock_guard<std::mutex> g(l->ex_mu); if (!l->jobs.empty()) { idle = false; break; } }
            if (!idle) break;
            for (int s = 0; s < P.H; s++) {
                if (s == r) continue;
                char *hl = P.ring_hdr(s, r);
                if (vld64(hl) != X.rx_tail[s]) { idle = false; break; }
            }
        }
        stable = idle ? stable + 1 : 0;
        if (!idle) _mm_pause();
    }
}
Cache::Budget Cache::owner_budget(int o) {
    Host &X = *p_->hosts[o];
    std::lock_guard<std::mutex> g(X.mu);
    p_->expire_ts_charge(X);
    return {cfg_.budget_bytes, cfg_.charge_value_bytes_only || cfg_.charge_model == CHARGE_K11B ? 0 : cfg_.ai_slots * sizeof(AiEnt),
            X.obj_bytes, X.li_bytes,
            X.ts_bytes, X.objs.size()};
}
void Cache::emu_prefetch(int h, int o, uint64_t off) {
    Host &X = *p_->hosts[h];
    std::array<char, 64> a;
    load64(p_->slab(o) + off, a.data());
    std::lock_guard<std::mutex> g(X.emu_mu);
    X.emu[(uint64_t)o << 40 | off] = a;
}
void Cache::emu_drop_all(int h) {
    Host &X = *p_->hosts[h];
    std::lock_guard<std::mutex> g(X.emu_mu);
    X.emu.clear();
}
std::string Cache::describe() const {
    char b[1024];
    snprintf(b, sizeof b,
             "scheme=%s hosts=%d budget_bytes=%zu slab_bytes=%zu ai_slots=%zu ring_lines=%zu heat_batch=%d max_probe=%d "
             "shm_node=%d bind_hosts=%d emu_noncoherent=%d rules: tomb=%d bump2=%d pending=%d filltok=%d "
             "flush_before_install=%d slow_deadline=%d unpublish_on_put=%d put_cut_at_publish=%d swap_refresh_after_del=%d swap_del_first=%d slow_limit_ns=%lu li_entry_bytes=%u "
             "ts_entry_bytes=%u shm_bytes=%zu",
             scheme_name(cfg_.scheme), cfg_.hosts, cfg_.budget_bytes, cfg_.slab_bytes, cfg_.ai_slots, cfg_.ring_lines, cfg_.heat_batch,
             cfg_.max_probe, cfg_.shm_node, (int)cfg_.bind_hosts, (int)cfg_.emu_noncoherent, cfg_.rules.tomb,
             cfg_.rules.bump2, cfg_.rules.pending, cfg_.rules.filltok, cfg_.rules.flush_before_install,
             cfg_.rules.slow_deadline, cfg_.rules.unpublish_on_put, cfg_.rules.put_cut_at_publish, cfg_.rules.swap_refresh_after_del, cfg_.rules.swap_del_first, (unsigned long)cfg_.slow_limit_ns, cfg_.li_entry_bytes,
             cfg_.ts_entry_bytes, p_->shm_bytes);
    const Rules &r = cfg_.rules;
    std::string out = b;
    snprintf(b, sizeof b,
             " lease=%d revoke=%d lease_ns=%lu tick_ns=%lu lease_from_ai_read=%d lease_bracket=%d lease_tsc=%d lease_clock_hit=%d put_wait=%d put_wait_from_cut=%d "
             "put_publish=%d restart_wait=%d put_timeout_ns=%lu",
             r.lease, r.revoke, (unsigned long)cfg_.lease_ns, (unsigned long)p_->tick_ns(), r.lease_from_ai_read,
             r.lease_bracket, r.lease_tsc, r.lease_clock_hit, r.put_wait, r.put_wait_from_cut, r.put_publish, r.restart_wait, (unsigned long)cfg_.put_timeout_ns);
    out += b;
    snprintf(b, sizeof b,
             " update=%s log=%d log_poll_ns=%lu log_interval_ns=%lu log_wait_ns=%lu log_lines=%zu log_sweep_slots=%zu "
             "gen_bits=%d li_slots=%zu restamp=%d put_serial=%d log_ts_before_tails=%d log_wait_from_cut=%d log_hb_before=%d log_seqlock=%d log_hit_clock=%d log_install_check=%d "
             "log_snap_before=%d log_unpublish_first=%d log_every_put=%d log_sweep=%d log_dropall_lag=%d log_overrun=%d "
             "log_epoch=%d feed=%d feed_del=%d refresh_flush=%d feed_restamp=%d kg_cxl=%d kg_early=%d kg_buckets=%zu "
             "li_cap_factor=%g owners=%d reuse_delayed=%d log_batch_tails=%d log_sweep_per_poll=%lu log_prefetch_lines=%u log_rec8=%d "
             "xtra_dir_slots_per_owner=%zu li16=%d li16_fp_bits=%d li16_ep_bits=%d log_sweep_pass_s=%.3g log_sweep_margin=%.3g log_bloom_bits=%u log_refresh_hot_only=%d hit_clock=%s obj_read=%s obj_prefetch_min_lines=%u xtra_own_freshness=%d xtra_hk_ns=%lu",
             update_name(r), r.log, (unsigned long)cfg_.log_poll_ns, (unsigned long)p_->log_interval(),
             (unsigned long)p_->log_W_ns(), cfg_.log_lines, cfg_.log_sweep_slots, cfg_.gen_bits, cfg_.li_slots, r.restamp,
             r.put_serial, r.log_ts_before_tails, r.log_wait_from_cut, r.log_hb_before, r.log_seqlock, r.log_hit_clock, r.log_install_check, r.log_snap_before,
             r.log_unpublish_first, r.log_every_put, r.log_sweep, r.log_dropall_lag, r.log_overrun, r.log_epoch, r.feed,
             r.feed_del, r.refresh_flush, r.feed_restamp, r.kg_cxl, r.kg_early, cfg_.kg_buckets, cfg_.li_cap_factor,
             cfg_.owners, (int)cfg_.reuse_delayed, (int)cfg_.log_batch_tails, (unsigned long)p_->sweep_slots(),
             cfg_.log_prefetch_lines, (int)cfg_.log_rec8, p_->xdir_mask ? p_->xdir_mask + 1 : (size_t)0, (int)p_->li16_on(),
             cfg_.li16_fp_bits, cfg_.li16_ep_bits,
             Cache::Impl::sweep_pass_s(cfg_), cfg_.log_max_alloc_per_s > 0 ? Cache::Impl::sweep_margin_of(cfg_) : -1.0,
             cfg_.log_bloom_bits, (int)cfg_.log_refresh_hot_only, cfg_.hit_clock == 1 ? "rdtscp" : "lfence3",
             obj_read_mode_name(p_->cfg.obj_read_mode), cfg_.obj_prefetch_min_lines, (int)p_->xtra_own(),
             (unsigned long)cfg_.xtra_hk_ns);
    out += b;
    if (cfg_.xtra_dir_replicas) out += " xtra_dir_replicas=" + std::to_string(cfg_.xtra_dir_replicas);   // printed only when on
    snprintf(b, sizeof b, " charge=%s exec_lanes=%d exec_timer_slack_ns=%lu put_async=%d put_spin_ns=%lu put_idle_ns=%lu handler_idle_us=%u put_key_after=%s%s",
             cfg_.charge_value_bytes_only ? "value" : cfg_.charge_model == CHARGE_K11B ? "k11b" : "layout", p_->exec_lanes_eff(),
             (unsigned long)cfg_.exec_timer_slack_ns, (int)p_->put_async_on(), (unsigned long)cfg_.put_spin_ns,
             (unsigned long)cfg_.put_idle_ns, cfg_.handler_idle_us, cfg_.put_key_after_publish ? "publish" : "reply",
             cfg_.ro_peers ? " ro_peers=1" : "");   // printed only when on
    out += b;
    snprintf(b, sizeof b, " li_clock_inq=%d%s", (int)cfg_.li_clock_inq, RC_MONO ? " rc_mono=1" : "");   // RC_MONO build: shown
    if (cfg_.mono) {   // Config::mono: only when on
        char m[160];
        snprintf(m, sizeof m, " mono=1 mono_fd_ns=%lu mono_hold_ns=%lu mono_buckets=%zu mono_hold_rule=%d", (unsigned long)cfg_.mono_fd_ns,
                 (unsigned long)(cfg_.rules.mono_hold ? cfg_.mono_fd_ns + cfg_.log_poll_ns : 0), cfg_.mono_buckets, (int)cfg_.rules.mono_hold);
        return out + b + m;
    }
    return out + b;
}

// ---------------------------------------------------------------- Client
Client::Client(Cache *c, int host) : c_(c), host_(host) {
    id_ = c->p_->new_mailbox(host);
    heat_.resize(c->cfg_.hosts);
}
Client::~Client() {
    flush_heat();
    flush_stats();
}
ClientStats Client::totals() {
    flush_stats();
    return total_;
}
void Client::flush_stats() {
    Stats &S = *c_->p_->hosts[host_]->st;
#define F(x) if (cs_.x) { S.x += cs_.x; total_.x += cs_.x; cs_.x = 0; }
    F(heat_reported) F(slow_noentry) F(slow_logdel) F(slow_marked) F(tok_timeout) F(shard_remote) F(shard_overrun) F(shard_overrun_miss) F(get_absent) F(get_fast) F(get_slow) F(get_miss) F(fast_fail) F(slow_fail) F(ai_absent) F(lease_expired) F(lease_retry) F(hb_stale) F(dropgen_miss) F(log_refused) F(heat_applied) F(heat_stale) F(slow_timeout) F(tomb_refuse) F(ts_reclaimed) F(put_failed) F(xdir_reads) F(xdir_other) F(xdir_stale) F(mono_table) F(mono_backend) F(mono_records) F(mono_overflow) F(mono_ovf_backend) F(mono_bk_moved) F(mono_bk_kg) F(mono_bk_slow) F(mono_table_miss)
#undef F
    for (int k = 0; k < 3; k++)
        for (int m = 0; m < 5; m++)
            if (cs_.shm[k][m]) { S.shm[k][m] += cs_.shm[k][m]; total_.shm[k][m] += cs_.shm[k][m]; cs_.shm[k][m] = 0; }
    nstat_ = 0;
}

void Client::flush_heat() {
    Cache::Impl &P = *c_->p_;
    for (int o = 0; o < P.H; o++) {
        auto &v = heat_[o];
        size_t i = 0;
        while (i < v.size()) {
            uint32_t n = (uint32_t)std::min<size_t>(v.size() - i, 4 * 31);   // up to 31 data lines
            std::vector<HeatEnt> e(n);
            for (uint32_t j = 0; j < n; j++) e[j] = {v[i + j].first, v[i + j].second, 0};
            MsgHdr m{};
            m.type = M_HEAT; m.nlines = 1; m.a = n;
            P.send(host_, o, &m, e.data(), n * sizeof(HeatEnt), true);
            i += n;
        }
        v.clear();
    }
}

// lease_bracket: a hit whose GET spanned >= 2 epochs (it may have read lines of two generations after a pause of
// a full counter round, or used an AI entry read long ago) is discarded and the GET retried.
uint32_t Client::get(uint64_t key, char *out, Kind *kind, uint32_t *ver_out) {
    const ShmCount shm0 = tl_shm;
    uint32_t v = get_body(key, out, kind, ver_out);
    uint64_t *b = cs_.shm[*kind == FAST ? 0 : *kind == SLOW ? 1 : 2];
    b[0] += tl_shm.flush - shm0.flush; b[1] += tl_shm.rd - shm0.rd; b[2] += tl_shm.wr - shm0.wr; b[3] += tl_shm.msg - shm0.msg;
    b[4]++;
    return v;
}
uint32_t Client::get_body(uint64_t key, char *out, Kind *kind, uint32_t *ver_out) {
    Cache::Impl &P = *c_->p_;
    if (++nstat_ >= 1024) flush_stats();
    if (!P.lease_scheme() || !P.cfg.rules.lease_bracket) return get_once(key, out, kind, ver_out);
    Host &X = *P.hosts[host_];
    for (int attempt = 0;; attempt++) {
        uint32_t e_start = P.epoch_now(X);
        uint32_t vlen = get_once(key, out, kind, ver_out);
        if (*kind == MISS) return vlen;   // value straight from the backend
        uint32_t e_end = P.epoch_now(X);
        if (e_end - e_start <= 1) return vlen;
        cs_.lease_retry++;
        if (attempt >= 8) {   // give up on the cache: read the backend
            uint32_t ver = P.be->get(key, out, &vlen);
            if (ver_out) *ver_out = ver;
            *kind = MISS;
            return vlen;
        }
    }
}

#if RC_MONO
// Config::mono, out of line (cold): a hit of a key in the table -- AI[k] (flush + load, after the object's lines) must still name the
// copy (slot + generation) with kg >= the need; returns -1 (serve the hit) or why the GET reads the backend instead: 0 = the
// bucket overflowed, 1 = AI[k] no longer names the copy, 2 = the copy's kg < the need (3 = a slow path's, see get_once)
__attribute__((noinline, cold)) static int mono_hit_check(Cache::Impl &P, int o, uint64_t key, uint64_t off, uint32_t gen,
                                                         uint32_t need, int st) {
    if (st == 2) return 0;
    Cache::Impl::AiView av;
    if (P.ai_lookup(o, key, &av) != 1 || av.off != off || av.gen != gen) return 1;
    if (av.kg < need) return 2;
    return -1;
}
__attribute__((noinline, cold)) uint32_t Client::mono_backend_read(int o, uint64_t key, char *out, Kind *kind, uint32_t *ver_out,
                                                                    int why) {
    Cache::Impl &P = *c_->p_;
    Host &X = *P.hosts[host_];
    cs_.mono_backend++;
    if (why == 0) cs_.mono_ovf_backend++;
    else if (why == 1) cs_.mono_bk_moved++;
    else if (why == 2) cs_.mono_bk_kg++;
    else cs_.mono_bk_slow++;
    uint32_t vlen = 0;
    uint32_t ver = P.be->get(key, out, &vlen);
    P.mono_record(X, key, P.kg_load(o, key) + 1, cs_);
    cs_.get_miss++;
    *kind = MISS;
    if (ver_out) *ver_out = ver;
    return vlen;
}
#endif
uint32_t Client::get_once(uint64_t key, char *out, Kind *kind, uint32_t *ver_out) {
    Cache::Impl &P = *c_->p_;
    Host &X = *P.hosts[host_];
    const Config &cfg = P.cfg;
    int o = c_->owner_of(key);
    if (cfg.heat_batch > 0 && ++nreq_ % cfg.heat_batch == 0) flush_heat();
    uint32_t vlen = 0;
    // monotonic reads (Config::mono): the host's table, read ONCE at the GET's start and kept; a GET of a key in
    // it whose cached copy does not pass reads the backend (no install, no fill) and records the kg loaded after the data read
#if RC_MONO
    int mono_st = 0;
    uint32_t mono_fl = 0;
    bool mono_counted = false;
    if (__builtin_expect(cfg.mono, 0)) mono_st = P.mono_snap(X, key, &mono_fl);
#endif
    // slot: the local entry of a fast hit (heat_dedup), nullptr for a slow-path hit (always reported)
    auto hit = [&](uint64_t off, uint32_t gen, const LiRef *slot = nullptr) {
        if (o == host_) { P.apply_heat(o, off, gen, &cs_); return; }   // owner's own hit: set the visited bit directly
        if (cfg.heat_batch <= 0) return;
        if (slot && cfg.heat_dedup) {   // at most one report per entry per window (benign race: heat is a hint)
            if (slot->seen_window(P.heat_window(X))) return;
        }
        heat_[o].emplace_back(off, gen);   // non-owner: batched report
        cs_.heat_reported++;
    };
    const Scheme sc = cfg.scheme;
    if (sc == Scheme::SHARD && o != host_) {   // ---- SHARD, remote key: GET RPC to the owner
        Mailbox dmb, *mb = &dmb;   // direct mode: no mailbox (P.mailbox takes a host-wide mutex)
        if (!cfg.shard_direct) mb = P.mailbox(host_, id_);
        uint64_t t0 = mono_ns();
        if (cfg.shard_direct) {   // shard_direct: read the owner's private cache directly (threads mode), token too on a miss
            uint32_t dv = 0;
            bool dh = P.priv_get(o, key, out, &dv);
            mb->val2 = dh ? 1 : 0;
            mb->val = dh ? dv : P.priv_token(o, key);
        } else {
            mb->buf = out;
            mb->ready.store(0);
            MsgHdr m{};
            m.type = M_SGET; m.nlines = 1; m.key = key; m.a = id_;
            P.send(host_, o, &m, nullptr, 0, false);
            P.wait_mailbox(mb);
        }
        uint64_t pad = cfg.rpc_ns;
        if (cfg.rpc_dist) {   // sampled round trip for this reply size (miss reply: a 64 B header)
            if (!rng_) rng_ = 0x9E3779B97F4A7C15ULL * (uint64_t)(host_ + 1) + (uint64_t)id_ * 0xBF58476D1CE4E5B9ULL + 1;
            pad = rpc_dist_sample(&cfg.rpc_dist->d, mb->val2 == 1 ? std::max<uint32_t>(mb->val, 64) : 64, &rng_);
        }
        cs_.shard_remote++;
        if (pad && mono_ns() - t0 >= pad) {   // the channel (or the direct read) itself was slower than the target
            cs_.shard_overrun++;
            if (mb->val2 != 1) cs_.shard_overrun_miss++;
        }
        while (pad && mono_ns() - t0 < pad) _mm_pause();   // pad to the measured RPC round trip
        if (mb->val2 == 1) {
            vlen = mb->val;
            cs_.get_slow++;   // remote hit
            *kind = SLOW;
            return vlen;
        }
        uint32_t tok = mb->val;
        hook(Hook::MissAfterToken, host_, key);
        uint32_t ver = P.be->get(key, out, &vlen);
        hook(Hook::MissBeforeSendFill, host_, key);
        if (vlen) {
            MsgHdr f{};
            f.type = M_SFILL; f.nlines = 1; f.key = key; f.a = tok; f.b = vlen; f.c = ver;
            P.send(host_, o, &f, out, vlen, true);
        } else {   // deleted (backend tombstone): nothing to fill
            cs_.get_absent++;
            hook(Hook::FillDone, host_, key);
        }
        cs_.get_miss++;
        *kind = MISS;
        if (ver_out) *ver_out = ver;
        return vlen;
    }
    if (sc == Scheme::PRIV || sc == Scheme::SHARD) {   // ---- PRIV: private cache in this host's DRAM; SHARD: own key, served locally
        if (P.priv_get(host_, key, out, &vlen)) {
            cs_.get_fast++;
            *kind = FAST;
            return vlen;
        }
        uint32_t tok = P.priv_token(host_, key);
        hook(Hook::MissAfterToken, host_, key);
        uint32_t ver = P.be->get(key, out, &vlen);
        hook(Hook::MissBeforeSendFill, host_, key);
        if (vlen) P.priv_fill(host_, key, tok, out, vlen, ver);
        else cs_.get_absent++;   // deleted (backend tombstone): nothing to fill
        hook(Hook::FillDone, host_, key);
        cs_.get_miss++;
        *kind = MISS;
        if (ver_out) *ver_out = ver;
        return vlen;
    }
    if (sc == Scheme::GSTATE_CXL || sc == Scheme::GSTATE_COH) {   // ---- one shared state word per GET
        const char *w = P.owner_base(o) + P.off_gstate;
        if (sc == Scheme::GSTATE_CXL) {
            flush_line(w);
            _mm_mfence();
        }
        tl_shm.rd++;
        if (vld64(w) == 0xdeadbeefdeadbeefULL) cs_.heat_stale++;   // value feeds a (never taken) branch
    }
    if (sc == Scheme::VER || sc == Scheme::VER_NOFLUSH) {   // ---- no local index: AI entry, then the object
        Cache::Impl::AiView a;
        if (P.ai_lookup(o, key, &a, sc == Scheme::VER) == 1) {
            if (P.obj_read(host_, o, a.off, a.gen, key, a.nl, out, &vlen) ||
                (_mm_mfence(), P.obj_flush(host_, o, a.off, a.nl),   // flush after the reads (see the slow path)
                 P.obj_read(host_, o, a.off, a.gen, key, a.nl, out, &vlen))) {
                hit(a.off, a.gen);
                cs_.get_fast++;
                *kind = FAST;
                return vlen;
            }
            cs_.slow_fail++;
        } else {
            cs_.ai_absent++;
        }
        goto miss;
    }
    {
    // 1. fast path: local index, one 64 B load per line; no shared writes, no shared metadata reads
    //    (log: plus one host-local heartbeat load before the lookup and one TSC read after the loads)
    const bool logm = P.log_scheme();
    const bool xown = P.xtra_own();   // XTRA own freshness: freshness from the directory word, no heartbeat
    uint64_t hb0 = 0;
    if (logm && !xown && cfg.rules.log_hb_before) hb0 = X.hb.load(std::memory_order_acquire);
    LiEnt e;
    uint64_t lv = 0;
    uint32_t le0 = 0;
    LiRef lslot;
    bool have = X.li.find(key, o, &lv, &le0, &lslot);   // lock-free
    last_reason_ = R_NONE;
    if (!have) { cs_.slow_noentry++; last_reason_ = R_NOENTRY; }
    else if (!(uint32_t)lv && (lv >> 63)) { cs_.slow_marked++; have = false; last_reason_ = R_MARKED; }   // failed before
    else if (!(uint32_t)lv) { cs_.slow_logdel++; have = false; last_reason_ = R_LOGDEL; }   // log: deleted by a record
    if (have && P.lease_scheme() && P.epoch_now(X) - le0 > 1) {   // lease expired
        cs_.lease_expired++;
        have = false;
        last_reason_ = R_LEASE;
    }
    if (have && logm && le0 != X.li.ep_of(X.lv[o].drop_gen.load(std::memory_order_acquire))) {   // dropped with all of o's entries
        cs_.dropgen_miss++;
        have = false;
        last_reason_ = R_DROPALL;
    }
    if (have) e = {(lv >> 32 & ((1ULL << 28) - 1)) * LINE, (uint32_t)lv, CLS[lv >> 60]};
    if (have) hook(Hook::FastAfterLiRead, host_, key);
    if (have) {
        if (sc == Scheme::CXL_UC) P.obj_flush(host_, o, e.off, e.nl);   // no CPU-cache reuse
        // own freshness (xtra_own) compares the key's slot version with the version the entry was installed under (CD versions);
        // the borrowed version only counts the comparison of the offset's word with the object's generation
        bool xok = true;
        if (sc == Scheme::XTRA && xown) {
            xok = P.xver_load(o, host_, key, cs_) == e.gen;
            if (!xok) cs_.xdir_other++;
        } else if (sc == Scheme::XTRA) {
            xok = P.xdir_check(o, host_, e.off, e.gen, cs_);
        }
        uint32_t ogen = 0;   // own freshness: the object's generation, from its line 0 (the entry holds the slot version)
        if (xown && !xok) {   // own freshness: the key's slot moved since the entry was installed: revalidate (slow path)
            cs_.xdir_stale++;
            last_reason_ = R_XDIR;
            Stripe &s = P.stripe(host_, key);
            std::lock_guard<std::mutex> g(s.mu);
            X.li.erase(key, o, lv);   // only if it still holds that entry
        } else if (P.obj_read(host_, o, e.off, xown ? 0 : e.gen, key, e.nl, out, &vlen, xown ? &ogen : nullptr)) {
            hook(Hook::FastAfterLoad, host_, key);
#if RC_MONO
            if (__builtin_expect(mono_st != 0, 0)) {   // mono: AI[k] (read after the object) must still name this copy with kg > the floor
                cs_.mono_table++;
                mono_counted = true;
                const int why = mono_hit_check(P, o, key, e.off, e.gen, mono_fl, mono_st);
                if (why >= 0) return mono_backend_read(o, key, out, kind, ver_out, why);
            }
#endif
            if (!logm || xown || P.hb_ok(X, hb0)) {
                const uint32_t g = xown ? ogen : e.gen;
                hit(e.off, g, &lslot);
                // local-index cap: a read marks the entry (only if clear: threads hitting a hot key do not all write its slot)
                if (!lslot.ref()) lslot.set_ref();
                last_o_ = o; last_off_ = e.off; last_gen_ = g;
                cs_.get_fast++;
                *kind = FAST;
                return vlen;
            }
            cs_.hb_stale++;   // poller late or stalled: the entry may be stale; take the slow path
            last_reason_ = R_HBSTALE;
        } else {
            cs_.fast_fail++;
            last_reason_ = R_GENMISS;
            Stripe &s = P.stripe(host_, key);
            std::lock_guard<std::mutex> g(s.mu);
            if (P.feed_on() && cfg.rules.keep_failed) X.li.mark_failed(key, o, lv);   // still held: a refresh repoints it
            else X.li.erase(key, o, lv);   // only if it still holds the entry that failed
        }
    }
    // 2. slow path: flush + read the AI entry -> flush the object's lines -> tombstone check + install -> read.
    //    Flushing BEFORE the install keeps other threads of this host from using the new gen while a stale
    //    (possibly half-written) instance of the line is still in this host's cache.
    //    A slow path that took longer than T (slow_limit_ns) by the time it installs is abandoned and restarted;
    //    together with tombstones living T after the revoke this bounds tombstone lifetime.
    if (logm) {
        // 2'. log slow path: s0 = records applied (and the drop generation) BEFORE reading the AI entry; flush
        //     the object's lines; install only if no record for the key was applied since s0 (check + install atomic
        //     w.r.t. the poller: poll_mu). Not installing still returns the value read (the GET may return it).
        LogView &L = X.lv[o];
        uint64_t s0 = 0;
        uint32_t dg0 = 0;
        const bool dt = cfg.diag_lock_timing;   // diag_lock_timing: whole slow path and the poll_mu wait
        const uint64_t dt0 = dt ? __rdtsc() : 0;
        struct DiagEnd {
            Stats *st; uint64_t t0; bool on;
            ~DiagEnd() { if (on) diag_add(*st, 1, __rdtsc() - t0); }
        } diag_end{X.st, dt0, dt};
        if (cfg.rules.log_snap_before) {
            s0 = L.applied.load(std::memory_order_acquire);
            dg0 = L.drop_gen.load(std::memory_order_acquire);
        }
        // CD versions (own freshness): the key's slot version BEFORE the AI entry and the object, kept in the entry: a publish
        // after this read moves the version, so the entry fails its next hit once (reading it only after the AI entry is unsafe)
        const uint32_t vseen = xown ? P.xver_load(o, host_, key, cs_) : 0;
        Cache::Impl::AiView a;
        int r = P.ai_lookup(o, key, &a);
        hook(Hook::SlowAfterAiRead, host_, key);
        if (!cfg.rules.log_snap_before) {   // ablation: s0 after the AI read; test variant log-snapafter (G3-installchk) fails
            s0 = L.applied.load(std::memory_order_acquire);
            dg0 = L.drop_gen.load(std::memory_order_acquire);
        }
        if (r != 1) {
            cs_.ai_absent++;
#if RC_MONO
        } else if (__builtin_expect(mono_st != 0, 0) && (mono_st == 2 || a.kg < mono_fl)) {   // mono: the AI's object is not newer than the floor
            if (!mono_counted) cs_.mono_table++;
            return mono_backend_read(o, key, out, kind, ver_out, mono_st == 2 ? 0 : 3);
#endif
        } else {
            _mm_mfence();   // flush after reading the entry (see the lease slow path below)
            if (cfg.rules.flush_before_install) P.obj_flush(host_, o, a.off, a.nl);
            {
                const uint64_t w0 = dt ? __rdtsc() : 0;
                std::lock_guard<std::mutex> pg(X.poll_mu);
                if (dt) diag_add(*X.st, 0, __rdtsc() - w0);
                if ((cfg.rules.log_install_check &&
                     (L.rec_seq[Cache::Impl::rec_bucket(key)] > s0 || L.drop_gen.load(std::memory_order_relaxed) != dg0)) ||
                    (xown && !vseen)) {   // (version 0: the slot was never published when read; not installed)
                    cs_.log_refused++;
                } else {
                    Stripe &s = P.stripe(host_, key);
                    const uint64_t w1 = dt ? __rdtsc() : 0;
                    std::lock_guard<std::mutex> g(s.mu);
                    if (dt) diag_add(*X.st, 3, __rdtsc() - w1);
                    uint64_t v0;
                    bool held = X.li.find(key, o, &v0);
                    uint64_t at = 0;
                    if (X.li.put(key, o, li_pack(a.off, xown ? vseen : a.gen, class_of(a.nl)), dg0, true, &at) && !held && cfg.li_cap_factor > 0) {
                        if (!cfg.li_clock_inq) {   // li_clock_inq off: every install of an entry not held pushes
                            if (X.li.c16) X.clock_qi.push_back((uint32_t)at);   // under poll_mu
                            else X.clock_q.push_back(key);
                        } else if (X.li.c16) {   // Config::li_clock_inq: push only if the slot is not queued yet
                            if (X.clock_inq_i.size() <= at) X.clock_inq_i.resize((size_t)X.li.b.mask + 1, 0);
                            if (!X.clock_inq_i[at]) { X.clock_inq_i[at] = 1; X.clock_qi.push_back((uint32_t)at); }
                        } else if (X.clock_inq_k.insert(key).second) {
                            X.clock_q.push_back(key);
                        }
                    }
                }
            }
            hook(Hook::SlowAfterInstall, host_, key);
            if (!cfg.rules.flush_before_install) P.obj_flush(host_, o, a.off, a.nl);
            if (sc == Scheme::XTRA && !xown) P.xdir_check(o, host_, a.off, a.gen, cs_);   // own: the slot was read first (above)
            if (P.obj_read(host_, o, a.off, a.gen, key, a.nl, out, &vlen)) {
                hit(a.off, a.gen);
                last_o_ = o; last_off_ = a.off; last_gen_ = a.gen;
#if RC_MONO
                if (__builtin_expect(cfg.mono, 0)) {   // mono: a slow-path return: floor = the returned object's kg - 1, i.e. need = its kg
                    if (mono_st && !mono_counted) cs_.mono_table++;
                    P.mono_record(X, key, a.kg, cs_);
                }
#endif
                cs_.get_slow++;
                *kind = SLOW;
                return vlen;
            }
            cs_.slow_fail++;
        }
        goto miss;
    }
    for (int attempt = 0; attempt < 3; attempt++) {
        uint64_t t0 = mono_ns();
        const bool lease = P.lease_scheme();
        if (lease && cfg.rules.lease_tsc) P.epoch_check(X);
        uint32_t e0 = P.epoch_now(X);   // lease starts BEFORE the AI entry is read
        Cache::Impl::AiView a;
        int r = P.ai_lookup(o, key, &a);
        hook(Hook::SlowAfterAiRead, host_, key);
        if (r != 1) { cs_.ai_absent++; break; }
        // The object's line addresses come from the entry just loaded (an address dependency), but x86 does not
        // architecturally order CLFLUSHOPT after older loads (SDM: ordered only by fences, locked RMWs and older
        // writes to the same line). The mfence makes "flush after reading the entry" explicit.
        _mm_mfence();
        if (cfg.rules.flush_before_install) P.obj_flush(host_, o, a.off, a.nl);
        int verdict;   // 0 adopted, 1 refused by a tombstone, 2 too old: restart
        {
            Stripe &s = P.stripe(host_, key);
            std::lock_guard<std::mutex> g(s.mu);
            uint64_t now = mono_ns();
            auto it = s.ts.find(key);
            if (it != s.ts.end() && now - it->second.t > cfg.slow_limit_ns) {
                s.ts.erase(it);
                it = s.ts.end();
                cs_.ts_reclaimed++;
            }
            if (cfg.rules.slow_deadline && now - t0 > cfg.slow_limit_ns) verdict = 2;
            else if (cfg.rules.tomb && it != s.ts.end() && a.kg < it->second.kg) verdict = 1;
            else {
                if (lease && !cfg.rules.lease_from_ai_read) e0 = X.epoch.load(std::memory_order_acquire);   // ablation: L1-atinstall
                X.li.put(key, o, li_pack(a.off, a.gen, class_of(a.nl)), e0);
                verdict = 0;
            }
        }
        if (verdict == 2) { cs_.slow_timeout++; continue; }
        if (verdict == 1) { cs_.tomb_refuse++; break; }
        hook(Hook::SlowAfterInstall, host_, key);
        if (!cfg.rules.flush_before_install) P.obj_flush(host_, o, a.off, a.nl);
        if (sc == Scheme::XTRA) P.xdir_check(o, host_, a.off, a.gen, cs_);
        if (P.obj_read(host_, o, a.off, a.gen, key, a.nl, out, &vlen)) {
            hit(a.off, a.gen);
            last_o_ = o; last_off_ = a.off; last_gen_ = a.gen;
            cs_.get_slow++;
            *kind = SLOW;
            return vlen;
        }
        cs_.slow_fail++;
        break;
    }
    }
miss:
    // 3. miss: fill token from the owner -> backend -> fill to the owner (admitted asynchronously). An owner that does
    //    not answer within tok_timeout_ns: read the backend, no fill, and skip it for tok_backoff_ns (its keys stay
    //    readable; nobody but the owner ever fills or writes them).
    //    kg_cxl: the token is the owner's kg bucket word, flush-loaded from CXL BEFORE the backend read; no message.
    uint32_t tok = 0;
    bool fill = true;
    const bool kgc = P.kg_cxl();
    if (kgc) {
        if (cfg.rules.kg_early) tok = P.kg_load(o, key);
    } else if (o == host_) {
        std::lock_guard<std::mutex> g(X.mu);
        tok = P.kg_of(X, key);
    } else {
        if (tok_skip_until_.empty()) tok_skip_until_.assign(P.H, 0);
        uint64_t now = mono_ns();
        if (now < tok_skip_until_[o]) {
            fill = false;
        } else {
            Mailbox *mb = P.mailbox(host_, id_);
            uint32_t seq = mb->seq.fetch_add(1) + 1;
            if (seq == 0) seq = mb->seq.fetch_add(1) + 1;
            mb->ready.store(0);
            MsgHdr m{};
            m.type = M_TOKREQ; m.nlines = 1; m.key = key; m.a = id_; m.d = seq;
            P.send(host_, o, &m, nullptr, 0, false);
            tok = P.wait_mailbox(mb, cfg.tok_timeout_ns);
            if (!mb->ready.load(std::memory_order_acquire)) {
                mb->seq.fetch_add(1);   // a late token for this request is dropped
                tok_skip_until_[o] = mono_ns() + cfg.tok_backoff_ns;
                fill = false;
            }
        }
        if (!fill) cs_.tok_timeout++;
    }
    hook(Hook::MissAfterToken, host_, key);
    uint32_t ver = P.be->get(key, out, &vlen);
#if RC_MONO
    if (__builtin_expect(cfg.mono, 0)) {   // mono: a backend return: floor = the key's kg loaded AFTER the data read (need = floor + 1)
        if (mono_st && !mono_counted) { cs_.mono_table++; cs_.mono_table_miss++; }
        P.mono_record(X, key, P.kg_load(o, key) + 1, cs_);
    }
#endif
    hook(Hook::MissBeforeSendFill, host_, key);
    if (kgc && !cfg.rules.kg_early) tok = P.kg_load(o, key);   // ablation: token after the backend read (test F3-kglate)
    if (!vlen) {   // the key is deleted (the backend's tombstone): a miss without a fill
        cs_.get_absent++;
        hook(Hook::FillDone, host_, key);
    } else if (!fill) {
        hook(Hook::FillDone, host_, key);
    } else if (o == host_) {   // owner's own miss: the fill is counted as a fill (no message), not in this GET
        const ShmCount s0 = tl_shm;
        P.owner_fill(o, key, tok, out, vlen, ver);
        shm_add(X.st->shm[SHM_FILL], s0);
        tl_shm = s0;
        hook(Hook::FillDone, host_, key);
    } else {
        MsgHdr m{};
        m.type = M_FILL; m.nlines = 1; m.key = key; m.a = tok; m.b = vlen; m.c = ver;
        P.send(host_, o, &m, out, vlen, true);
    }
    cs_.get_miss++;
    *kind = MISS;
    if (ver_out) *ver_out = ver;
    return vlen;
}

uint32_t Client::put(uint64_t key) { return write_op(key, W_PUT); }
uint32_t Client::del(uint64_t key) { return write_op(key, W_DEL); }
uint32_t Client::expire(uint64_t key) { return write_op(key, W_EXPIRE); }
uint32_t Client::write_op(uint64_t key, int op) {
    Cache::Impl &P = *c_->p_;
    if (++nstat_ >= 1024) flush_stats();
    if (P.cfg.heat_batch > 0 && ++nreq_ % P.cfg.heat_batch == 0) flush_heat();
    int o = c_->owner_of(key);
    if (put_skip_until_.empty()) put_skip_until_.assign(P.H, 0);
    if (mono_ns() < put_skip_until_[o]) {   // its owner timed out recently: fail at once (nobody else writes the key)
        cs_.put_failed++;
        return 0;
    }
    Mailbox *mb = P.mailbox(host_, id_);
    uint32_t seq = mb->seq.fetch_add(1) + 1;
    if (seq == 0) seq = mb->seq.fetch_add(1) + 1;
    mb->ready.store(0);
    const uint64_t t_send = P.cfg.diag_put_timing ? __rdtsc() : 0;
    if (o == host_) {
        Host &X = *P.hosts[o];
        Host::Lane &L = P.lane_of(X, key);
        std::lock_guard<std::mutex> g(L.ex_mu);
        L.jobs.push_back({key, host_, id_, seq, t_send, t_send, op});
        L.njobs.fetch_add(1, std::memory_order_release);
        L.ex_cv.notify_one();
    } else {
        MsgHdr m{};
        m.type = M_PUTREQ; m.nlines = 1; m.key = key; m.a = id_; m.c = (uint32_t)op; m.d = seq; m.e = t_send;
        P.send(host_, o, &m, nullptr, 0, false);
    }
    // all PUTs go through the owner; if it does not answer, the PUT fails (nobody else writes the backend)
    uint32_t ver = P.wait_mailbox(mb, P.cfg.put_timeout_ns);
    if (!mb->ready.load(std::memory_order_acquire)) {
        mb->seq.fetch_add(1);   // a late PUTDONE for this request is dropped
        cs_.put_failed++;
        put_skip_until_[o] = mono_ns() + P.cfg.put_backoff_ns;
        return 0;
    }
    if (t_send) {   // diag_put_timing: reply transit and the whole PUT, on the writer's host
        uint64_t now = __rdtsc();
        if (mb->stamp) diag_put_add(*P.hosts[host_]->st, 8, mb->stamp, now);
        diag_put_add(*P.hosts[host_]->st, 9, t_send, now);
    }
    if (P.cfg.scheme == Scheme::PRIV && op == W_PUT) {   // write = access at the writer: install the value it wrote
        char buf[MAX_OUT];
        uint32_t vlen;
        P.be->value_of(key, ver, buf, &vlen);
        uint32_t tok = mb->val2;   // token captured when the PUT completed
        hook(Hook::PrivWriterBeforeInstall, host_, key);
        if (!P.cfg.rules.priv_writer_token) tok = P.priv_token(host_, key);
        P.priv_fill(host_, key, tok, buf, vlen, ver);
    }
    return ver;
}

}  // namespace rc
