// Synthetic load: H hosts x C client threads, Zipf keys, GETs and PUTs, every GET checked independently.
//   default (threads mode): all hosts are threads of this process.
//   --procs: one process per host. This process creates the shared area (POSIX shm, NUMA-bound) and a
//     second shm segment for the simulated backend, the checker's completed-PUT register and per-host stats,
//     forks one child per host (pinned to NUMA node h with --bind), reports, and cleans up.
// Optional chaos: at every protocol hook point, sleep a random 0-200 us with probability p (widens races).
// Blast radius (--procs): --fault stop-reader:H | kill-reader:H | stop-owner:H stops (SIGSTOP, SIGCONT after
// --fault-len s) or kills (SIGKILL) host H's process at --fault-at s; --owners K makes hosts K.. pure readers; --reuse
// delayed is the PCCIndex-style baseline. --series <csv> writes per host and per 100 ms: GETs and their latency, PUTs,
// failed PUTs and their latency, checker errors, and the owner's admissions, budget use and pending reclaims.
// Exit status 0 iff the checker saw no error.
#include <fcntl.h>
#include <getopt.h>
#include <numa.h>
#include <sys/mman.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "rackcache.h"
#include "sim.h"

using namespace rc;

static void usage() {
    fprintf(stderr,
            "usage: stress [--procs] [--scheme rackcache|cxl_uc|ver|ver_noflush|gstate_cxl|gstate_coh|priv|shard|xtra] [--rpc-ns 0] [--hosts 4] [--clients 3] [--keys 100000] [--zipf 0.99] [--put 0.05] [--del 0] [--expire 0] [--ro-peers 0|1 (other hosts' blocks mapped read-only, --procs)] [--mono 0|1 (monotonic reads; P6 checked)] [--mono-hold 0|1] [--mono-fd-us 10000] [--put-order window|swap] [--put-wait 1|0 (swap + 0: committed, threads)]\n"
            "  [--budget-mb 16] [--slab-mb 32] [--ai-slots 262144] [--ring 4096] [--heat 16] [--delay-ns 2000]\n"
            "  [--seconds 60] [--chaos 0] [--seed 1] [--shm-node -1] [--bind] [--slow-limit-us 10000]\n"
            "  [--no-tomb] [--no-bump2] [--no-pending] [--no-deadline] [--no-unpublish]\n"
            "  [--update log|lease|invalidate]   (default: log; legacy: --revoke, --no-lease)\n"
            "  [--lease-us 1000] [--tick-us T/2] [--clock-hit]   (lease)\n"
            "  [--poll-us 1000] [--interval-us P/2] [--log-wait-us P] [--log-lines 4096] [--sweep-slots 1024] [--gen-bits 32]   (log)\n"
            "  [--report 10] [--owners K] [--reuse immediate|delayed] [--reuse-timeout-ms 0] [--revoke-timeout-ms 0 (invalidation: hosts that do not ack are declared dead)]\n"
            "  [--emu (software non-coherent reader caches)] [--emu-prefetch p (per GET: one random slab line into the cache)]\n"
            "  [--heat-dedup 1|0] [--heat-window-polls 1] [--shard-direct (threads mode, scheme shard)] [--keep-failed 0|1]\n"
            "  [--prefetch-lines 8 (record lines read at once)] [--rec8 1|0 (1 = 8 B delete records, 0 = 16 B)]\n"
            "  [--xtra-dir-slots 1048576] [--xtra-own-freshness 0|1] [--xtra-dir-replicas 0|1|2 (a directory copy per host; 1 clwb, 2 clflushopt)] [--li32 (32 B local-index entries; 16 B is the default, --li16 is a no-op) [--li16-fp-bits 20] [--li16-ep-bits 16]] [--hit-clock lfence3|rdtscp] [--obj-read copy|prefetch|mstore|both] [--put-async 0|1] [--put-key-after reply|publish]\n"
            "  [--max-alloc-rate 1.22e6 (per owner, for the R2 sweep-margin check; 0 = no check)]\n"
            "  [--bloom-bits 8 (Bloom bits per local-index slot; 0 = off)] [--li-slots 1048576 (power of two)]\n"
            "  [--refresh-hot-only 0|1 (a refresh of an unread entry only deletes it; default 0)]\n"
            "  [--feed 1|0 (1 change feed / 0 lazy log)] [--kg cxl|msg] [--li-cap 1 (x objects; 0 = unlimited)] [--kg-buckets 65536]\n"
            "  [--put-timeout-ms 2000] [--series out.csv] [--fault stop-reader:H|kill-reader:H|stop-owner:H --fault-at 10 --fault-len 60]   (faults need --procs)\n");
    exit(2);
}

static thread_local uint64_t tl_rng = 0;
static double g_emu_prefetch = 0;
static double g_tsc_per_ms = 1;
static double tsc_per_ms() {   // for the poller CPU time in the report
    uint64_t t0 = now_ns(), c0 = __rdtsc();
    while (now_ns() - t0 < 5'000'000) _mm_pause();
    return (double)(__rdtsc() - c0) / ((double)(now_ns() - t0) / 1e6);
}
static double g_chaos = 0;
// --put-order swap --put-wait 0 = Krater's committed ack (threads mode): the checker counts bounded staleness from
// the cut (err_stale), P2 misses are info (raw_misses), as replay / tests
static bool g_committed = false;
static void chaos_hook(Hook, int, uint64_t) {
    uint64_t r = splitmix(tl_rng);
    if ((r >> 11) * (1.0 / 9007199254740992.0) < g_chaos) {
        uint64_t us = splitmix(tl_rng) % 200;
        std::this_thread::sleep_for(std::chrono::microseconds(us));
    }
}

constexpr int MAXH = 16;
// ---- time series (--series): per host, per 100 ms slot (shared anonymous mapping, so forked hosts add into it at the end)
constexpr uint64_t SER_NS = 100'000'000;
constexpr int SER_MAX = 3000;   // 300 s
struct SerSlot {
    std::atomic<uint64_t> gets{0}, get_ns{0}, puts{0}, put_ns{0}, put_fail{0}, errs{0};
    std::atomic<uint32_t> gh[40], ph[40];   // log2(ns) histograms
};
struct SerLocal {
    uint64_t gets = 0, get_ns = 0, puts = 0, put_ns = 0, put_fail = 0, errs = 0;
    uint32_t gh[40] = {0}, ph[40] = {0};
};
static SerSlot *g_ser = nullptr;
static std::atomic<uint64_t> *g_ser_t0 = nullptr;
static inline int lg2(uint64_t ns) { return ns ? std::min(39, 64 - __builtin_clzll(ns)) : 0; }
struct SimShared {   // head of the simulator segment (--procs); then ver[nkeys+1], done[nkeys+1]
    std::atomic<int> ready{0}, stop{0}, clients_done{0}, failed{0}, killed_clients{0};
    CheckCounters cc;
    Stats stats[MAXH];
};

struct Workload {
    std::vector<double> cdf;
    std::vector<uint64_t> perm;
    double putp;
    double delp = 0, expp = 0;   // DELETE / EXPIRE shares (drawn from the same number as PUTs: 0 = none)
    uint64_t seed;
    uint64_t key(uint64_t &rng) const {
        double u = (splitmix(rng) >> 11) * (1.0 / 9007199254740992.0);
        return perm[std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin()];
    }
};

static void client_loop(Cache &cache, Checker &ck, const Workload &w, int h, int c, const std::atomic<int> &stop,
                        bool bind) {
    if (bind && numa_available() >= 0) numa_run_on_node(h % (numa_max_node() + 1));
    tl_rng = w.seed * 1000003 + h * 101 + c;
    uint64_t rng = w.seed * 0x1234567ULL + h * 1000 + c;
    Client cl(&cache, h);
    std::vector<char> buf(MAX_OUT);
    std::vector<SerLocal> ser(g_ser ? SER_MAX : 0);
    while (!stop.load(std::memory_order_relaxed)) {
        uint64_t key = w.key(rng);
        double q = (splitmix(rng) >> 11) * (1.0 / 9007199254740992.0);
        uint64_t t0 = g_ser ? now_ns() : 0;
        bool put = q < w.putp, ok = true;
        const bool del = !put && q < w.putp + w.delp, exp = !put && !del && q < w.putp + w.delp + w.expp;
        uint32_t v = 0;
        if (g_emu_prefetch > 0 && (splitmix(rng) >> 11) * (1.0 / 9007199254740992.0) < g_emu_prefetch) {
            // --emu-prefetch: a "prefetcher" pulls a random line of a random owner's slab into this host's (non-coherent) cache
            int o = (int)(splitmix(rng) % (uint64_t)cache.cfg().hosts);
            uint64_t off = (splitmix(rng) % (cache.cfg().slab_bytes / LINE)) * LINE;
            cache.emu_prefetch(h, o, off);
        }
        if (put) {
            v = cl.put(key);
            ck.put_done(key, v);
        } else if (del) {   // the tombstone's version completes like a PUT's
            v = cl.del(key);
            ck.put_done(key, v);
        } else if (exp) {   // TTL: no version
            v = cl.expire(key);
        } else {
            uint32_t req = ck.begin(key);
            const uint32_t sfl = g_committed ? ck.stale_begin(key) : 0;   // committed: the version bounded staleness demands
            const uint32_t mf = ck.mono_begin(h, key);   // P6 (--mono 0|1): this host's highest returned version so far
            Kind kind;
            uint32_t vlen = cl.get(key, buf.data(), &kind);
            ok = ck.verify(key, req, buf.data(), vlen, h, kind) || g_committed;
            if (sfl) { uint32_t gv; memcpy(&gv, buf.data() + 8, 4); ok &= ck.verify_stale(key, sfl, gv, h, kind); }
            if (ck.mono_on()) { uint32_t gv; memcpy(&gv, buf.data() + 8, 4); ck.verify_mono(h, key, mf, gv, kind); }
        }
        if (!g_ser) continue;
        uint64_t t1 = now_ns(), base = g_ser_t0->load(std::memory_order_relaxed);
        if (!base || t1 < base || (t1 - base) / SER_NS >= (uint64_t)SER_MAX) continue;
        SerLocal &x = ser[(t1 - base) / SER_NS];
        if (!put && !del && !exp) { x.gets++; x.get_ns += t1 - t0; x.gh[lg2(t1 - t0)]++; x.errs += !ok; }   // writes: puts
        else if (v) { x.puts++; x.put_ns += t1 - t0; x.ph[lg2(t1 - t0)]++; }
        else x.put_fail++;
    }
    for (int i = 0; i < (int)ser.size(); i++) {
        SerLocal &x = ser[i];
        if (!x.gets && !x.puts && !x.put_fail) continue;
        SerSlot &y = g_ser[h * SER_MAX + i];
        y.gets += x.gets; y.get_ns += x.get_ns; y.puts += x.puts; y.put_ns += x.put_ns; y.put_fail += x.put_fail; y.errs += x.errs;
        for (int b = 0; b < 40; b++) { if (x.gh[b]) y.gh[b] += x.gh[b]; if (x.ph[b]) y.ph[b] += x.ph[b]; }
    }
}
// owner-side gauges sampled by the launcher every 100 ms
struct OwnerSample { uint64_t admits, used, limit, pending, dead, fill_drop_space; };
static double hq(const std::atomic<uint32_t> *b, uint64_t n, double p) {   // quantile of a log2 histogram (bucket top)
    if (!n) return 0;
    uint64_t want = (uint64_t)std::ceil(p * n), c = 0;
    for (int i = 0; i < 40; i++) if ((c += b[i]) >= want) return std::ldexp(1.0, i);
    return std::ldexp(1.0, 39);
}
static void write_series(const std::string &path, int H, int nslots, const std::vector<std::vector<OwnerSample>> &os,
                         double fault_at, double fault_len, int fault_host) {
    FILE *f = fopen(path.c_str(), "w");
    if (!f) { perror(path.c_str()); return; }
    fprintf(f, "t_s,host,fault_active,gets,get_ns_mean,get_ns_p50,get_ns_p99,puts,put_ns_mean,put_ns_p99,put_failed,errors,"
               "admits,owner_used,owner_limit,reclaim_pending,dead_hosts,fill_drop_space\n");
    for (int i = 0; i < nslots && i < SER_MAX; i++)
        for (int h = 0; h < H; h++) {
            SerSlot &y = g_ser[h * SER_MAX + i];
            double t = (i + 1) * SER_NS / 1e9;
            bool fa = fault_host >= 0 && t > fault_at && (fault_len <= 0 || t <= fault_at + fault_len + 1e-9);
            const OwnerSample *a = i < (int)os.size() ? &os[i][h] : nullptr, *b = i > 0 && i - 1 < (int)os.size() ? &os[i - 1][h] : nullptr;
            fprintf(f, "%.1f,%d,%d,%lu,%.0f,%.0f,%.0f,%lu,%.0f,%.0f,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", t, h, fa ? 1 : 0,
                    (unsigned long)y.gets.load(), y.gets ? (double)y.get_ns / y.gets : 0, hq(y.gh, y.gets, 0.5), hq(y.gh, y.gets, 0.99),
                    (unsigned long)y.puts.load(), y.puts ? (double)y.put_ns / y.puts : 0, hq(y.ph, y.puts, 0.99),
                    (unsigned long)y.put_fail.load(), (unsigned long)y.errs.load(),
                    (unsigned long)(a ? a->admits - (b ? b->admits : 0) : 0), (unsigned long)(a ? a->used : 0),
                    (unsigned long)(a ? a->limit : 0), (unsigned long)(a ? a->pending : 0), (unsigned long)(a ? a->dead : 0),
                    (unsigned long)(a ? a->fill_drop_space - (b ? b->fill_drop_space : 0) : 0));
        }
    fclose(f);
}

static void report(double el, int H, const std::function<Stats &(int)> &st, const std::string &ck) {
    auto total = [&](auto f) { uint64_t s = 0; for (int h = 0; h < H; h++) s += f(st(h)); return s; };
#define T(x) (unsigned long)total([](Stats &s) { return s.x.load(); })
    uint64_t f = T(get_fast), sl = T(get_slow), m = T(get_miss), g = f + sl + m;
    printf("t=%.0fs gets=%lu fast=%.1f%% slow=%.1f%% miss=%.1f%% puts=%lu evict=%lu admits=%lu fast_fail=%lu slow_fail=%lu "
           "tomb_refuse=%lu slow_timeout=%lu ts_reclaimed=%lu fill_drop[tok=%lu pend=%lu cached=%lu space=%lu] "
           "ring_drop=%lu overflow=%lu heat[ok=%lu stale=%lu reported=%lu] msgs=%lu lease[expired=%lu retry=%lu epoch_stall=%lu "
           "tick_late=%lu tick_late_max_us=%lu] put_failed=%lu log[records=%lu polls=%lu applied=%lu deleted=%lu refused=%lu "
           "hb_stale=%lu dropall=%lu overrun=%lu sweep=%lu restamps=%lu restamp_ovf=%lu poll_ms=%.1f refreshed=%lu li_evicted=%lu bloom_skip=%lu bloom_rebuilds=%lu refresh_cold=%lu] xdir[reads=%lu other=%lu stale=%lu] | %s\n",
           el, (unsigned long)g, g ? 100.0 * f / g : 0, g ? 100.0 * sl / g : 0, g ? 100.0 * m / g : 0, T(puts),
           T(evictions), T(admits), T(fast_fail), T(slow_fail), T(tomb_refuse), T(slow_timeout), T(ts_reclaimed),
           T(fill_drop_tok), T(fill_drop_pending), T(fill_drop_cached), T(fill_drop_space), T(drop_ring_full),
           T(overflowed), T(heat_applied), T(heat_stale), T(heat_reported), T(msgs_sent), T(lease_expired), T(lease_retry), T(epoch_stall),
           T(tick_late), [&] { uint64_t m = 0; for (int h = 0; h < H; h++) m = std::max<uint64_t>(m, st(h).tick_late_max_ns); return (unsigned long)(m / 1000); }(),
           T(put_failed), T(log_records), T(polls), T(log_applied), T(log_deleted), T(log_refused), T(hb_stale), T(log_dropall),
           T(log_overrun), T(sweep_dropped), T(restamps), T(restamp_overflow), (double)total([](Stats &s) { return s.poll_cycles.load(); }) / g_tsc_per_ms,
           T(log_refreshed), T(li_evicted), T(log_bloom_skip), T(bloom_rebuilds), T(log_refresh_cold), T(xdir_reads), T(xdir_other), T(xdir_stale),
           ck.c_str());
    if (T(deletes) + T(expires))   // printed only when the run executed DELETEs / EXPIREs
        printf("  writes: deletes=%lu expires=%lu get_absent=%lu\n", T(deletes), T(expires), T(get_absent));
#undef T
    fflush(stdout);
}

// P6 (--mono 0|1): monotonic-read errors and Config::mono's counters, summed over the hosts
static void mono_report(bool on, int H, const std::function<Stats &(int)> &st, uint64_t errs, uint64_t checked) {
    uint64_t t = 0, tm = 0, b = 0, bm = 0, bk = 0, bs = 0, r = 0, o = 0, ob = 0, fd = 0, f = 0, sl = 0;
    for (int h = 0; h < H; h++) {
        Stats &x = st(h);
        t += x.mono_table; tm += x.mono_table_miss; b += x.mono_backend; bm += x.mono_bk_moved; bk += x.mono_bk_kg; bs += x.mono_bk_slow;
        r += x.mono_records; o += x.mono_overflow; ob += x.mono_ovf_backend; fd += x.mono_fd_overrun; f += x.get_fast; sl += x.get_slow;
    }
    // would-be hits = fast + slow + the backend reads Config::mono took instead of a hit (it only does so at a hit / slow check)
    const double wb = (double)(f + sl + b);
    printf("MONO mono=%d mono_errors=%lu gets_checked=%lu | table_gets=%lu (plain misses %lu) table_share_of_would_be_hits=%.6f "
           "backend_instead=%lu (%.6f of would-be hits; moved=%lu kg=%lu slow=%lu overflow=%lu) records=%lu overflow=%lu fd_overrun=%lu\n",
           (int)on, (unsigned long)errs, (unsigned long)checked, (unsigned long)t, (unsigned long)tm, wb ? (double)(t - tm) / wb : 0.0,
           (unsigned long)b, wb ? (double)b / wb : 0.0, (unsigned long)bm, (unsigned long)bk, (unsigned long)bs, (unsigned long)ob,
           (unsigned long)r, (unsigned long)o, (unsigned long)fd);
}
static void print_budget(Cache &cache, int o) {
    Cache::Budget b = cache.owner_budget(o);
    printf("# owner %d: budget=%lu = ai_table %lu + objects %lu + li %lu + tombstones %lu + free %ld  (objects=%lu)\n", o,
           (unsigned long)b.limit, (unsigned long)b.ai_table, (unsigned long)b.objects, (unsigned long)b.li,
           (unsigned long)b.ts, (long)b.limit - (long)(b.ai_table + b.objects + b.li + b.ts), (unsigned long)b.nobj);
}

static void sample_owners(std::vector<std::vector<OwnerSample>> &os, uint64_t t0, int H, const std::function<Stats &(int)> &st) {
    while ((uint64_t)(os.size() + 1) * SER_NS <= now_ns() - t0) {   // one sample at the end of every 100 ms slot
        std::vector<OwnerSample> v(H);
        for (int h = 0; h < H; h++) {
            Stats &x = st(h);
            v[h] = {x.admits.load(), x.owner_used.load(), x.owner_limit.load(), x.reclaim_pending.load(), x.dead_hosts.load(),
                    x.fill_drop_space.load()};
        }
        os.push_back(v);
    }
}

int main(int argc, char **argv) {
    Config cfg;
    cfg.budget_bytes = 16 << 20;   // includes the AI table (4 MiB at 2^18 slots), LI entries and tombstones
    cfg.slab_bytes = 32 << 20;
    cfg.ai_slots = 1 << 18;
    cfg.ring_lines = 1 << 12;
    int clients = 3, report_s = 10, fault_host = -1, mono_arg = -1;
    bool put_order_arg = false;
    std::string series, fault;
    double fault_at = 10, fault_len = 60;
    bool procs = false;
    uint64_t nkeys = 100000, delay_ns = 2000;
    double zipf = 0.99, seconds = 60;
    Workload w;
    w.putp = 0.05;
    w.seed = 1;
    static option opts[] = {{"hosts", 1, 0, 'H'}, {"clients", 1, 0, 'c'}, {"keys", 1, 0, 'k'}, {"zipf", 1, 0, 'z'},
                            {"put", 1, 0, 'p'}, {"del", 1, 0, 61}, {"expire", 1, 0, 62}, {"ro-peers", 1, 0, 63}, {"mono", 1, 0, 1064}, {"mono-hold", 1, 0, 1065}, {"mono-fd-us", 1, 0, 1066}, {"put-order", 1, 0, 1067}, {"put-wait", 1, 0, 1068}, {"budget-mb", 1, 0, 'b'}, {"slab-mb", 1, 0, 'S'}, {"ai-slots", 1, 0, 'a'},
                            {"ring", 1, 0, 'r'}, {"heat", 1, 0, 'h'}, {"delay-ns", 1, 0, 'd'}, {"seconds", 1, 0, 't'},
                            {"chaos", 1, 0, 'x'}, {"seed", 1, 0, 's'}, {"shm-node", 1, 0, 'n'}, {"bind", 0, 0, 'B'},
                            {"no-tomb", 0, 0, 1}, {"no-bump2", 0, 0, 2}, {"no-pending", 0, 0, 3}, {"report", 1, 0, 'R'},
                            {"no-deadline", 0, 0, 4}, {"no-unpublish", 0, 0, 5}, {"slow-limit-us", 1, 0, 'T'},
                            {"procs", 0, 0, 'P'}, {"scheme", 1, 0, 'C'}, {"rpc-ns", 1, 0, 'Q'},
                            {"revoke", 0, 0, 6}, {"no-lease", 0, 0, 7}, {"lease-us", 1, 0, 'L'}, {"tick-us", 1, 0, 'K'}, {"clock-hit", 0, 0, 8},
                            {"update", 1, 0, 'U'}, {"poll-us", 1, 0, 9}, {"interval-us", 1, 0, 10}, {"log-wait-us", 1, 0, 11},
                            {"log-lines", 1, 0, 12}, {"sweep-slots", 1, 0, 13}, {"gen-bits", 1, 0, 14},
                            {"owners", 1, 0, 15}, {"reuse", 1, 0, 16}, {"reuse-timeout-ms", 1, 0, 17}, {"series", 1, 0, 18},
                            {"fault", 1, 0, 19}, {"fault-at", 1, 0, 20}, {"fault-len", 1, 0, 21},
                            {"put-timeout-ms", 1, 0, 22}, {"feed", 1, 0, 23}, {"kg", 1, 0, 24}, {"li-cap", 1, 0, 25}, {"li-clock-inq", 1, 0, 1071},
                            {"kg-buckets", 1, 0, 26}, {"heat-dedup", 1, 0, 27}, {"heat-window-polls", 1, 0, 28},
                            {"shard-direct", 0, 0, 29}, {"keep-failed", 1, 0, 30}, {"revoke-timeout-ms", 1, 0, 31}, {"emu", 0, 0, 32}, {"emu-prefetch", 1, 0, 33},
                            {"prefetch-lines", 1, 0, 34}, {"rec8", 1, 0, 35}, {"xtra-dir-slots", 1, 0, 40}, {"xtra-own-freshness", 1, 0, 60}, {"xtra-dir-replicas", 1, 0, 64}, {"li16", 0, 0, 41},
                            {"li16-fp-bits", 1, 0, 42}, {"li16-ep-bits", 1, 0, 43}, {"max-alloc-rate", 1, 0, 36},
                            {"bloom-bits", 1, 0, 37}, {"li-slots", 1, 0, 39},
                            {"refresh-hot-only", 1, 0, 38}, {"li32", 0, 0, 44}, {"hit-clock", 1, 0, 45}, {"obj-read", 1, 0, 46}, {"put-async", 1, 0, 47}, {"put-key-after", 1, 0, 48}, {0, 0, 0, 0}};
    int ch;
    while ((ch = getopt_long(argc, argv, "", opts, nullptr)) != -1) {
        switch (ch) {
        case 'H': cfg.hosts = atoi(optarg); break;
        case 'c': clients = atoi(optarg); break;
        case 'k': nkeys = strtoull(optarg, 0, 0); break;
        case 'z': zipf = atof(optarg); break;
        case 'p': w.putp = atof(optarg); break;
        case 61: w.delp = atof(optarg); break;
        case 62: w.expp = atof(optarg); break;
        case 63: cfg.ro_peers = atoi(optarg) != 0; break;   // read-only peer blocks: needs --procs
        case 1064: mono_arg = atoi(optarg); cfg.mono = mono_arg == 1; break;   // Config::mono; if given (0 or 1), P6 is checked
        case 1065: cfg.rules.mono_hold = atoi(optarg) != 0; break;              // Config::mono ablation: 0 = entries expire at once
        case 1066: cfg.mono_fd_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 1067:   // as replay: window (default) | swap (Rules::put_cut_at_publish)
            if (std::string(optarg) == "swap") cfg.rules.put_cut_at_publish = true;
            else if (std::string(optarg) == "window") cfg.rules.put_cut_at_publish = false;
            else usage();
            put_order_arg = true;
            break;
        case 1068: cfg.rules.put_wait = atoi(optarg) != 0; put_order_arg = true; break;   // 0 with swap: the committed ack
        case 'b': cfg.budget_bytes = (size_t)(atof(optarg) * (1 << 20)); break;
        case 'S': cfg.slab_bytes = (size_t)(atof(optarg) * (1 << 20)); break;
        case 'a': cfg.ai_slots = strtoull(optarg, 0, 0); break;
        case 'r': cfg.ring_lines = strtoull(optarg, 0, 0); break;
        case 'h': cfg.heat_batch = atoi(optarg); break;
        case 'd': delay_ns = strtoull(optarg, 0, 0); break;
        case 't': seconds = atof(optarg); break;
        case 'x': g_chaos = atof(optarg); break;
        case 's': w.seed = strtoull(optarg, 0, 0); break;
        case 'n': cfg.shm_node = atoi(optarg); break;
        case 'B': cfg.bind_hosts = true; break;
        case 1: cfg.rules.tomb = false; break;
        case 2: cfg.rules.bump2 = false; break;
        case 3: cfg.rules.pending = false; break;
        case 4: cfg.rules.slow_deadline = false; break;
        case 5: cfg.rules.unpublish_on_put = false; break;
        case 6: cfg.rules.log = false; cfg.rules.revoke = true; break;   // --revoke: invalidation (revoke) instead of the log
        case 7: cfg.rules.lease = false; break;
        case 'U': if (!set_update(cfg.rules, optarg)) usage(); break;
        case 9: cfg.log_poll_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 10: cfg.log_interval_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 11: cfg.log_wait_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 12: cfg.log_lines = strtoull(optarg, 0, 0); break;
        case 13: cfg.log_sweep_slots = strtoull(optarg, 0, 0); break;
        case 14: cfg.gen_bits = atoi(optarg); break;
        case 15: cfg.owners = atoi(optarg); break;
        case 16: if (std::string(optarg) == "delayed") cfg.reuse_delayed = true; else if (std::string(optarg) != "immediate") usage(); break;
        case 17: cfg.reuse_timeout_ns = strtoull(optarg, 0, 0) * 1000000ULL; break;
        case 18: series = optarg; break;
        case 19: {
            std::string f = optarg;
            size_t c = f.find(':');
            if (c == std::string::npos) usage();
            fault = f.substr(0, c);
            fault_host = atoi(f.c_str() + c + 1);
            if (fault != "stop-reader" && fault != "kill-reader" && fault != "stop-owner") usage();
            break;
        }
        case 20: fault_at = atof(optarg); break;
        case 21: fault_len = atof(optarg); break;
        case 22: cfg.put_timeout_ns = strtoull(optarg, 0, 0) * 1000000ULL; break;
        case 23: cfg.rules.feed = atoi(optarg) != 0; break;
        case 24: if (std::string(optarg) == "cxl") cfg.rules.kg_cxl = true; else if (std::string(optarg) == "msg") cfg.rules.kg_cxl = false; else usage(); break;
        case 25: cfg.li_cap_factor = atof(optarg); break;
        case 1071: cfg.li_clock_inq = atoi(optarg) != 0; break;   // 0 = no in-queue mark: the dedup pass removes duplicates
        case 26: cfg.kg_buckets = strtoull(optarg, 0, 0); break;
        case 27: cfg.heat_dedup = atoi(optarg) != 0; break;
        case 28: cfg.heat_window_polls = (uint32_t)atoi(optarg); break;
        case 29: cfg.shard_direct = true; break;
        case 30: cfg.rules.keep_failed = atoi(optarg) != 0; break;
        case 31: cfg.revoke_timeout_ns = strtoull(optarg, 0, 0) * 1000000ULL; break;
        case 32: cfg.emu_noncoherent = true; break;
        case 33: g_emu_prefetch = atof(optarg); break;
        case 34: cfg.log_prefetch_lines = (uint32_t)atoi(optarg); break;
        case 35: cfg.log_rec8 = atoi(optarg) != 0; break;
        case 40: cfg.xtra_dir_slots = strtoull(optarg, 0, 0); break;
        case 60: cfg.xtra_own_freshness = atoi(optarg) != 0; break;   // XTRA-style own freshness
        case 64: {
            std::string v = optarg;
            if (v != "0" && v != "1" && v != "2") { fprintf(stderr, "--xtra-dir-replicas 0|1|2\n"); return 2; }
            cfg.xtra_dir_replicas = v[0] - '0';
            break;
        }
        case 41: cfg.li16 = true; break;
        case 42: cfg.li16_fp_bits = atoi(optarg); break;
        case 43: cfg.li16_ep_bits = atoi(optarg); break;
        case 44: cfg.li16 = false; break;
        case 46: if ((cfg.obj_read_mode = obj_read_mode_parse(optarg)) < 0) usage(); break;
        case 47: cfg.put_async = atoi(optarg) != 0; break;   // asynchronous PUT executor
        case 48:
            if (!strcmp(optarg, "publish")) cfg.put_key_after_publish = true;
            else if (!strcmp(optarg, "reply")) cfg.put_key_after_publish = false;
            else usage();
            break;
        case 45: if (std::string(optarg) == "rdtscp") cfg.hit_clock = 1; else if (std::string(optarg) == "lfence3") cfg.hit_clock = 0; else usage(); break;
        case 36: cfg.log_max_alloc_per_s = atof(optarg); break;
        case 37: cfg.log_bloom_bits = (uint32_t)atoi(optarg); break;
        case 38: cfg.log_refresh_hot_only = atoi(optarg) != 0; break;
        case 39: cfg.li_slots = strtoull(optarg, 0, 0); break;
        case 'L': cfg.lease_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 'K': cfg.lease_tick_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 8: cfg.rules.lease_clock_hit = true; break;
        case 'T': cfg.slow_limit_ns = strtoull(optarg, 0, 0) * 1000; break;
        case 'R': report_s = atoi(optarg); break;
        case 'P': procs = true; break;
        case 'C': if (!scheme_parse(optarg, &cfg.scheme)) usage(); break;
        case 'Q': cfg.rpc_ns = strtoull(optarg, 0, 0); break;
        default: usage();
        }
    }
    if (cfg.hosts > MAXH) usage();
    if (fault_host >= cfg.hosts || (!fault.empty() && !procs)) { fprintf(stderr, "--fault needs --procs and a valid host\n"); return 2; }
    if (fault == "kill-reader") fault_len = 0;
    if (!series.empty()) {   // shared anonymous: survives fork, so the hosts' processes add into the same slots
        size_t b = sizeof(SerSlot) * MAXH * SER_MAX + 64;
        char *p = (char *)mmap(nullptr, b, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { perror("mmap series"); return 1; }
        g_ser_t0 = new (p) std::atomic<uint64_t>(0);
        g_ser = (SerSlot *)(p + 64);
        for (int i = 0; i < MAXH * SER_MAX; i++) new (&g_ser[i]) SerSlot;
    }
    g_tsc_per_ms = tsc_per_ms();
    // Zipf CDF over ranks; rank -> key is a fixed permutation so that hot keys spread over owners
    w.cdf.resize(nkeys);
    double sum = 0;
    for (uint64_t i = 0; i < nkeys; i++) w.cdf[i] = (sum += 1.0 / std::pow((double)(i + 1), zipf));
    for (auto &c : w.cdf) c /= sum;
    w.perm.resize(nkeys);
    for (uint64_t i = 0; i < nkeys; i++) w.perm[i] = i + 1;
    uint64_t ps = w.seed * 7919;
    for (uint64_t i = nkeys - 1; i > 0; i--) std::swap(w.perm[i], w.perm[splitmix(ps) % (i + 1)]);
    char head[256];
    snprintf(head, sizeof head, "mode=%s clients_per_host=%d keys=%lu zipf=%.2f put=%.3f delay_ns=%lu seconds=%.0f chaos=%g seed=%lu "
             "owners=%d reuse=%s reuse_timeout_ms=%lu fault=%s:%d fault_at=%g fault_len=%g",
             procs ? "procs" : "threads", clients, (unsigned long)nkeys, zipf, w.putp, (unsigned long)delay_ns, seconds,
             g_chaos, (unsigned long)w.seed, cfg.owners, cfg.reuse_delayed ? "delayed" : "immediate",
             (unsigned long)(cfg.reuse_timeout_ns / 1000000), fault.empty() ? "none" : fault.c_str(), fault_host, fault_at, fault_len);
    if (cfg.ro_peers && !procs) { fprintf(stderr, "--ro-peers 1 needs --procs (one process per host)\n"); return 2; }
    if (cfg.ro_peers) snprintf(head + strlen(head), sizeof head - strlen(head), " ro_peers=1");   // only when on
    if (mono_arg >= 0) snprintf(head + strlen(head), sizeof head - strlen(head), " mono=%d mono_hold=%d", (int)cfg.mono, (int)cfg.rules.mono_hold);
    g_committed = cfg.rules.put_cut_at_publish && !cfg.rules.put_wait && cfg.rules.log && !cfg.rules.lease && !cfg.rules.revoke &&
                  cfg.scheme != Scheme::PRIV && cfg.scheme != Scheme::SHARD && !(cfg.scheme == Scheme::XTRA && cfg.xtra_own_freshness);
    if (put_order_arg)
        snprintf(head + strlen(head), sizeof head - strlen(head), " put_order=%s put_wait=%d committed=%d",
                 cfg.rules.put_cut_at_publish ? "swap" : "window", (int)cfg.rules.put_wait, (int)g_committed);
    if (g_committed && procs) { fprintf(stderr, "--put-order swap --put-wait 0 (committed): threads mode only\n"); return 2; }
    if (w.delp > 0 || w.expp > 0)   // only with deletes or expires
        snprintf(head + strlen(head), sizeof head - strlen(head), " del=%.3f expire=%.3f", w.delp, w.expp);
    if (g_chaos > 0) set_hook(chaos_hook);
    const int H = cfg.hosts;

    if (!procs) {   // ------------------------------------------------------------ threads mode
        SimBackend be(nkeys, delay_ns);
        Checker ck(nkeys, &be);
        if (mono_arg >= 0) ck.enable_mono(H);
        cfg.on_publish = [&](uint64_t k, uint32_t v) { ck.on_publish(k, v); };
        if (g_committed) {
            ck.set_stale_bound(cfg.log_poll_ns, cfg.log_poll_ns / 50);
            cfg.on_cut = [&ck](uint64_t k, uint32_t v, uint64_t c) { ck.on_cut(k, v, c); };
        }
        Cache cache(cfg, &be);
        printf("# stress: %s %s\n", cache.describe().c_str(), head);
        fflush(stdout);
        cache.start();
        cache.wait_writable();
        std::atomic<int> stop{0};
        std::vector<std::thread> th;
        for (int h = 0; h < H; h++)
            for (int c = 0; c < clients; c++)
                th.emplace_back([&, h, c] { client_loop(cache, ck, w, h, c, stop, cfg.bind_hosts); });
        uint64_t t0 = now_ns(), next = t0 + (uint64_t)report_s * 1000000000ULL;
        if (g_ser) g_ser_t0->store(t0);
        auto st = [&](int h) -> Stats & { return cache.stats(h); };
        std::vector<std::vector<OwnerSample>> os;
        while (now_ns() - t0 < (uint64_t)(seconds * 1e9)) {
            if (now_ns() >= next) { report((now_ns() - t0) / 1e9, H, st, ck.summary()); next += (uint64_t)report_s * 1000000000ULL; }
            if (g_ser) sample_owners(os, t0, H, st);
            std::this_thread::sleep_for(std::chrono::milliseconds(g_ser ? 5 : 50));
        }
        stop = 1;
        for (auto &t : th) t.join();
        report((now_ns() - t0) / 1e9, H, st, ck.summary());
        if (g_ser) write_series(series, H, (int)os.size(), os, fault_at, fault_len, -1);
        cache.stop();
        set_hook(HookFn());
        for (int o = 0; o < H; o++) print_budget(cache, o);
        for (int h = 0; h < H; h++) {   // shared-memory accesses per operation (per op: flush / CXL line reads / writes / msgs)
            static const char *nm[SHM_N] = {"fast", "slow", "miss", "fill", "put", "poll"};
            Stats &S = cache.stats(h);
            printf("# host %d shm:", h);
            for (int k = 0; k < SHM_N; k++) {
                double n = S.shm[k][4].load() ? (double)S.shm[k][4].load() : 1.0;
                printf(" %s[ops=%lu flush=%.3f rd=%.3f wr=%.3f msg=%.3f]", nm[k], (unsigned long)S.shm[k][4].load(),
                       S.shm[k][0].load() / n, S.shm[k][1].load() / n, S.shm[k][2].load() / n, S.shm[k][3].load() / n);
            }
            printf(" poll_cpu_ms=%.1f\n", S.poll_cpu_ns.load() / 1e6);
        }
        for (auto &e : ck.first_errors())
            printf("# %s: %s\n", g_committed && e.find(": P2:") != std::string::npos ? "raw miss (committed ack: not counted)" : "error",
                   e.c_str());
        if (mono_arg >= 0) mono_report(cfg.mono, H, st, ck.mono_errors(), ck.counters().mono_gets);
        const bool mfail = cfg.mono && ck.mono_errors();   // --mono 1: P6 must hold too (--mono 0: the control, reported only)
        const uint64_t errs = ck.errors() - (g_committed ? ck.p2_errors() : 0);
        if (g_committed)
            printf("RESULT %s errors=%lu committed=1 raw_misses=%lu stale_errors=%lu\n", errs || mfail ? "FAIL" : "OK", (unsigned long)errs,
                   (unsigned long)ck.p2_errors(), (unsigned long)ck.stale_errors());
        else
            printf("RESULT %s errors=%lu\n", ck.errors() || mfail ? "FAIL" : "OK", (unsigned long)ck.errors());
        return errs != 0 || mfail;
    }

    // ---------------------------------------------------------------------------- one process per host
    std::string tag = "/rackcache-" + std::to_string(getpid());
    cfg.shm_name = tag + "-cache";
    std::string simname = tag + "-sim";
    size_t cache_bytes = Cache::shm_create(cfg);
    size_t sim_bytes = sizeof(SimShared) + 2 * (nkeys + 1) * sizeof(std::atomic<uint32_t>);
    int fd = shm_open(simname.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, (off_t)sim_bytes)) { perror("sim shm"); return 1; }
    char *simp = (char *)mmap(nullptr, sim_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (simp == MAP_FAILED) { perror("mmap sim"); return 1; }
    SimShared *S = new (simp) SimShared;
    auto *ver = (std::atomic<uint32_t> *)(simp + sizeof(SimShared));
    auto *done = ver + (nkeys + 1);
    for (uint64_t i = 0; i <= nkeys; i++) { new (&ver[i]) std::atomic<uint32_t>(0); new (&done[i]) std::atomic<uint32_t>(0); }
    cfg.shared_stats = S->stats;
    printf("# stress: hosts=%d cache_shm=%s (%zu bytes) sim_shm=%s (%zu bytes) %s\n", H, cfg.shm_name.c_str(), cache_bytes,
           simname.c_str(), sim_bytes, head);
    fflush(stdout);
    std::vector<pid_t> kids;
    for (int h = 0; h < H; h++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); S->stop = 1; break; }
        if (pid == 0) {   // ---- child: host h
            if (cfg.bind_hosts && numa_available() >= 0) {
                numa_run_on_node(h % (numa_max_node() + 1));
                numa_set_preferred(h % (numa_max_node() + 1));
            }
            SimBackend be(nkeys, delay_ns, ver);
            Checker ck(nkeys, &be, done, &S->cc);
            if (mono_arg >= 0) ck.enable_mono(H);
            Config c = cfg;
            c.local_host = h;
            c.on_publish = [&](uint64_t k, uint32_t v) { ck.on_publish(k, v); };
            int rc = 0;
            {
                Cache cache(c, &be);
                if (h == 0) { printf("# config: %s\n", cache.describe().c_str()); fflush(stdout); }
                cache.start();
                cache.wait_writable();
                S->ready++;
                while (S->ready.load() < H && !S->stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                std::vector<std::thread> th;
                for (int k = 0; k < clients; k++)
                    th.emplace_back([&, k] {
                        client_loop(cache, ck, w, h, k, S->stop, false);
                        S->clients_done++;
                    });
                for (auto &t : th) t.join();
                // phase 2: stop the handlers only after every host's clients are done (in-flight PUTs need acks)
                while (S->clients_done.load() < H * clients - S->killed_clients.load())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                cache.stop();
                print_budget(cache, h);
                for (auto &e : ck.first_errors()) printf("# error (host %d): %s\n", h, e.c_str());
                fflush(stdout);
            }
            _exit(rc);
        }
        kids.push_back(pid);
    }
    while (S->ready.load() < H && !S->stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    uint64_t t0 = now_ns(), next = t0 + (uint64_t)report_s * 1000000000ULL;
    if (g_ser) g_ser_t0->store(t0);
    auto st = [&](int h) -> Stats & { return S->stats[h]; };
    SimBackend dummy(1, 0);
    Checker view(nkeys, &dummy, done, &S->cc);   // only for summary()/errors()
    std::vector<std::vector<OwnerSample>> os;
    int fstate = 0;   // 0 before the fault, 1 during, 2 after
    bool killed = false;
    uint64_t err_at_fault = 0, gets_at_fault = 0, err_at_end_fault = 0, gets_at_end_fault = 0;
    while (now_ns() - t0 < (uint64_t)(seconds * 1e9)) {
        double el = (now_ns() - t0) / 1e9;
        if (now_ns() >= next) { report(el, H, st, view.summary()); next += (uint64_t)report_s * 1000000000ULL; }
        if (fault_host >= 0 && fstate == 0 && el >= fault_at) {
            killed = fault == "kill-reader";
            if (killed) S->killed_clients += clients;
            kill(kids[fault_host], killed ? SIGKILL : SIGSTOP);
            err_at_fault = view.errors(); gets_at_fault = S->cc.gets;
            printf("# fault: %s host %d at t=%.2fs\n", killed ? "SIGKILL" : "SIGSTOP", fault_host, el);
            fflush(stdout);
            fstate = 1;
        } else if (fstate == 1 && !killed && el >= fault_at + fault_len) {
            kill(kids[fault_host], SIGCONT);
            err_at_end_fault = view.errors(); gets_at_end_fault = S->cc.gets;
            printf("# fault: SIGCONT host %d at t=%.2fs (during the stop: gets_checked=%lu errors=%lu)\n", fault_host, el,
                   (unsigned long)(gets_at_end_fault - gets_at_fault), (unsigned long)(err_at_end_fault - err_at_fault));
            fflush(stdout);
            fstate = 2;
        }
        if (g_ser) sample_owners(os, t0, H, st);
        std::this_thread::sleep_for(std::chrono::milliseconds(g_ser || fault_host >= 0 ? 5 : 50));
    }
    if (fstate == 1 && !killed) {   // the run ended during the stop: resume it so that it can shut down
        kill(kids[fault_host], SIGCONT);
        err_at_end_fault = view.errors(); gets_at_end_fault = S->cc.gets;
    }
    S->stop = 1;
    int bad_exit = 0;
    uint64_t deadline = now_ns() + 120ULL * 1000000000ULL;   // a crashed child would block the others' phase 2
    for (size_t i = 0; i < kids.size();) {
        int status = 0;
        pid_t r = waitpid(kids[i], &status, WNOHANG);
        if (r == kids[i]) {
            bool expected = killed && (int)i == fault_host && WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL;
            if (!expected && (!WIFEXITED(status) || WEXITSTATUS(status))) bad_exit++;
            i++;
        } else if (now_ns() > deadline) {
            fprintf(stderr, "children did not finish in 120 s, killing them\n");
            for (pid_t p : kids) kill(p, SIGKILL);
            deadline = ~0ULL;
            bad_exit++;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    report((now_ns() - t0) / 1e9, H, st, view.summary());
    if (g_ser) write_series(series, H, (int)os.size(), os, fault_at, fault_len, fault_host);
    if (fault_host >= 0)
        printf("FAULT %s host=%d at=%g len=%g gets_checked_before=%lu errors_before=%lu gets_checked_during=%lu errors_during=%lu "
               "gets_checked_after=%lu errors_after=%lu\n", fault.c_str(), fault_host, fault_at, killed ? 0.0 : fault_len,
               (unsigned long)gets_at_fault, (unsigned long)err_at_fault,
               (unsigned long)((killed ? S->cc.gets.load() : gets_at_end_fault) - gets_at_fault),
               (unsigned long)((killed ? view.errors() : err_at_end_fault) - err_at_fault),
               (unsigned long)(killed ? 0 : S->cc.gets.load() - gets_at_end_fault),
               (unsigned long)(killed ? 0 : view.errors() - err_at_end_fault));
    Cache::shm_unlink(cfg.shm_name);
    shm_unlink(simname.c_str());
    uint64_t errs = view.errors();
    if (mono_arg >= 0) mono_report(cfg.mono, H, st, S->cc.err_mono.load(), S->cc.mono_gets.load());
    const bool mfail = cfg.mono && S->cc.err_mono.load();
    printf("RESULT %s errors=%lu bad_child_exits=%d\n", errs || bad_exit || mfail ? "FAIL" : "OK", (unsigned long)errs, bad_exit);
    return errs != 0 || bad_exit != 0 || mfail;
}
