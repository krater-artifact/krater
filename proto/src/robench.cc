// The GET hit cost with the other host's block mapped read-only (Config::ro_peers) or read-write, ONE PROCESS PER
// HOST, which is the only mode where the protection exists.
// Two host processes on a POSIX shm area (as replay / stress --procs): host 0 owns every measured key (the area is bound to
// --data-node; the owner's threads run on that node's CPUs), host 1 is the reader (its service threads on the reader's and
// the data's nodes minus the reader CPU, the measuring thread pinned to --reader-cpu, as hitbench). Setup: --hot-bytes of
// --vs B objects; host 0 GETs every key (miss + local fill: admitted), then host 1 GETs every key (slow path: installs its
// local entry). Measured: --gets uniformly random GETs on host 1 per rep, each timed lfence; rdtsc ... rdtscp; lfence, after
// one warm-up rep (rep 0); every 16th GET is checked by sim.h's Checker outside the timed region.
// --pairs N: N fresh process pairs (fresh shm area) per setting, the two settings alternating (ro off / on, then on / off, ...),
// so slow drift hits both; --reps reps in each.
// --scheme rackcache (the fast path: local index + the owner's object lines) | ver (no local index: every GET flushes and
// reads the AI entry, then the object: clflushopt on read-only pages on every GET) | xtra (XTRA-style own freshness: every hit
// flushes and loads the key's directory word in the owner's block).
// Output per rep: CSV,scheme,ro_peers,pair,rep,vs,hot_bytes,keys,gets,fast,slow,miss,all_median_ns,all_p99_ns,all_mean_ns,
//   hit_median_ns,hit_p10_ns,hit_p90_ns,hit_p99_ns,hit_mean_ns  (hit = the scheme's hit kind: FAST; slow / miss excluded)
// Without AVX-512F the functional build reads a line with two 32 B loads (H5 not held): only the relative ro / rw
// difference means anything there, not the absolute numbers.
// --vary mono: the two settings are Config::mono off / on instead of ro_peers (column 3 = mono); with
// --mono-fd-us F (Config::mono_fd_ns; the hold is F + P) and --mono-buckets N. The reader's install pass records every key
// (slow-path returns): with a hold longer than the run every measured hit is of a key in the table (the AI re-read); with
// --mono-fd-us 0 the entries are expired (hold = P = 1 ms) before the warm-up rep ends, so a hit pays only the table lookup.
// Five more columns per rep then: mono_table,mono_backend,mono_ovf_backend,mono_bk_moved,mono_bk_kg (the reader's counters
// over the rep).
#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <numa.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "rackcache.h"
#include "sim.h"

using namespace rc;

static uint64_t sm64(uint64_t &s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double tsc_ghz() {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t c0 = __rdtsc();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    uint64_t c1 = __rdtsc();
    double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    return (c1 - c0) / ns;
}
static void usage() {
    fprintf(stderr, "usage: robench --vs B --hot-bytes N --data-node D --reader-cpu C [--scheme rackcache|ver|xtra] [--gets 1000000]\n"
                    "               [--reps 5] [--pairs 3] [--seed 1] [--vary ro|mono] [--mono-fd-us F] [--mono-buckets N]\n");
    exit(2);
}
static bool node_cpus(int node, int except, cpu_set_t *s) {
    bitmask *bm = numa_allocate_cpumask();
    bool ok = numa_node_to_cpus(node, bm) == 0;
    if (ok)
        for (unsigned c = 0; c < bm->size; c++)
            if (numa_bitmask_isbitset(bm, c) && (int)c != except) CPU_SET(c, s);
    numa_free_cpumask(bm);
    return ok;
}
struct Sync {   // head of the simulator segment; then ver[maxkey + 1], done[maxkey + 1]
    std::atomic<int> owner_ready{0}, reader_done{0}, fail{0};
    CheckCounters cc;
};

int main(int argc, char **argv) {
    uint32_t vs = 600;
    uint64_t hot = 64 << 20, gets = 1000000, seed = 1;
    int data_node = 1, cpu = 40, reps = 5, pairs = 3;
    std::string scheme_s = "rackcache", vary = "ro";
    int64_t mono_fd_us = -1;
    uint64_t mono_buckets = 0;
    static option opts[] = {{"vs", 1, 0, 'v'}, {"hot-bytes", 1, 0, 'h'}, {"data-node", 1, 0, 'd'}, {"reader-cpu", 1, 0, 'c'},
                            {"gets", 1, 0, 'g'}, {"reps", 1, 0, 'r'}, {"pairs", 1, 0, 'p'}, {"seed", 1, 0, 's'},
                            {"scheme", 1, 0, 'S'}, {"vary", 1, 0, 'V'}, {"mono-fd-us", 1, 0, 'F'}, {"mono-buckets", 1, 0, 'B'},
                            {0, 0, 0, 0}};
    int ch;
    while ((ch = getopt_long(argc, argv, "", opts, nullptr)) != -1) {
        switch (ch) {
        case 'v': vs = atoi(optarg); break;
        case 'h': hot = strtoull(optarg, 0, 0); break;
        case 'd': data_node = atoi(optarg); break;
        case 'c': cpu = atoi(optarg); break;
        case 'g': gets = strtoull(optarg, 0, 0); break;
        case 'r': reps = atoi(optarg); break;
        case 'p': pairs = atoi(optarg); break;
        case 's': seed = strtoull(optarg, 0, 0); break;
        case 'S': scheme_s = optarg; break;
        case 'V': vary = optarg; break;
        case 'F': mono_fd_us = strtoll(optarg, 0, 0); break;
        case 'B': mono_buckets = strtoull(optarg, 0, 0); break;
        default: usage();
        }
    }
    if (vs < 16 || vs > MAX_VLEN || (scheme_s != "rackcache" && scheme_s != "ver" && scheme_s != "xtra") || (vary != "ro" && vary != "mono"))
        usage();
    const bool vm = vary == "mono";
    const char *lab = vm ? "mono" : "ro_peers";
    const int reader_node = numa_node_of_cpu(cpu);
    if (reader_node < 0) { fprintf(stderr, "FATAL: cpu %d offline?\n", cpu); return 4; }
    const uint32_t nl = obj_lines(vs);
    const uint64_t obj_bytes = (uint64_t)nl * LINE, nkeys = hot / obj_bytes;
    std::vector<uint64_t> keys;   // keys owned by host 0
    for (uint64_t k = 1; keys.size() < nkeys; k++)
        if (Cache::owner_of_static(k, 2) == 0) keys.push_back(k);
    const uint64_t maxkey = keys.back();
    std::vector<uint32_t> sizes(maxkey + 1, vs);
    Config cfg;
    cfg.hosts = 2;
    if (scheme_s == "ver") cfg.scheme = Scheme::VER;
    if (scheme_s == "xtra") { cfg.scheme = Scheme::XTRA; cfg.xtra_own_freshness = true; }
    cfg.shm_node = data_node;
    cfg.heat_batch = 16;
    uint64_t ai = 1;
    while (ai < 2 * nkeys) ai <<= 1;
    cfg.ai_slots = ai;
    uint64_t li = 1;
    while (li < 2 * nkeys) li <<= 1;
    cfg.li_slots = li;
    cfg.li_cap_factor = 0;   // every key of the hot set stays held (no CLOCK push-outs during the measurement)
    cfg.slab_bytes = ((nkeys * obj_bytes) * 5 / 4 + (64 << 20)) & ~(size_t)(LINE - 1);
    cfg.budget_bytes = 4 * cfg.slab_bytes;   // everything fits (k11b charges or layout: no evictions)
    cfg.ring_lines = 1 << 14;
    cfg.rules.restart_wait = false;
    if (mono_fd_us >= 0) cfg.mono_fd_ns = (uint64_t)mono_fd_us * 1000;
    if (mono_buckets) cfg.mono_buckets = mono_buckets;
    const double ghz = tsc_ghz();
    printf("# robench: scheme=%s vs=%u lines=%u hot=%lu keys=%lu data_node=%d reader_cpu=%d (node %d) gets=%lu reps=%d "
           "pairs=%d tsc_ghz=%.4f%s\n", scheme_s.c_str(), vs, nl, (unsigned long)hot, (unsigned long)nkeys, data_node, cpu,
           reader_node, (unsigned long)gets, reps, pairs, ghz,
#if defined(__AVX512F__)
           ""
#else
           " (NO AVX-512F: functional build, two 32 B loads per line -- relative ro / rw difference only)"
#endif
    );
    if (vm) printf("# vary=mono mono_fd_ns=%lu mono_buckets=%zu (hold = mono_fd_ns + log_poll_ns %lu)\n", (unsigned long)cfg.mono_fd_ns,
                   cfg.mono_buckets, (unsigned long)cfg.log_poll_ns);
    printf("CSV_HEADER,scheme,%s,pair,rep,vs,hot_bytes,keys,gets,fast,slow,miss,all_median_ns,all_p99_ns,all_mean_ns,"
           "hit_median_ns,hit_p10_ns,hit_p90_ns,hit_p99_ns,hit_mean_ns%s\n", lab, vm ? ",mono_table,mono_backend,mono_ovf_backend,mono_bk_moved,mono_bk_kg" : "");
    fflush(stdout);
    int bad = 0;
    for (int p = 0; p < pairs; p++)
        for (int j = 0; j < 2; j++) {
            const bool ro = (p % 2 == 0) ? j == 1 : j == 0;   // off, on | on, off | ... (vary mono: ro = mono on)
            Config c = cfg;
            if (vm) c.mono = ro;
            else c.ro_peers = ro;
            c.shm_name = "/rcrob-" + std::to_string(getpid()) + "-" + std::to_string(p * 2 + j);
            Cache::shm_create(c);
            const size_t sim_bytes = sizeof(Sync) + 2 * (maxkey + 1) * sizeof(std::atomic<uint32_t>);
            char *sp = (char *)mmap(nullptr, sim_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (sp == MAP_FAILED) { perror("mmap sim"); return 1; }
            Sync *S = new (sp) Sync;
            auto *ver = (std::atomic<uint32_t> *)(sp + sizeof(Sync)), *done = ver + (maxkey + 1);
            for (uint64_t i = 0; i < 2 * (maxkey + 1); i++) new (&ver[i]) std::atomic<uint32_t>(0);
            fflush(stdout);
            pid_t kid[2];
            for (int h = 0; h < 2; h++) {
                kid[h] = fork();
                if (kid[h]) continue;
                // ---- host process h
                cpu_set_t svc; CPU_ZERO(&svc);
                if (h == 0) node_cpus(data_node, cpu, &svc);
                else { node_cpus(reader_node, cpu, &svc); node_cpus(data_node, cpu, &svc); }
                if (sched_setaffinity(0, sizeof svc, &svc)) { perror("sched_setaffinity"); _exit(4); }
                numa_set_preferred(h == 0 ? data_node : reader_node);
                SimBackend be(maxkey, 0, ver);
                be.set_sizes(sizes.data());
                Checker ck(maxkey, &be, done, &S->cc);
                Config hc = c;
                hc.local_host = h;
                hc.on_publish = [&](uint64_t k, uint32_t v) { ck.on_publish(k, v); };
                Cache cache(hc, &be);
                if (h == 1 && p == 0 && j == 0) { printf("# config: %s\n", cache.describe().c_str()); fflush(stdout); }
                cache.start();
                cache.wait_writable();
                std::vector<char> buf(MAX_OUT);
                Kind kind;
                if (h == 0) {   // owner: admit every key, then serve until the reader is done
                    // (an owner's own miss fills in the calling thread, so the keys are cached when get() returns;
                    // Cache::wait_idle() walks every host's state and works in one process only)
                    Client o(&cache, 0);
                    for (uint64_t k : keys) o.get(k, buf.data(), &kind);
                    uint64_t missing = 0;
                    for (uint64_t k : keys) if (!cache.owner_cached(k)) { o.get(k, buf.data(), &kind); missing++; }
                    for (uint64_t k : keys) if (!cache.owner_cached(k)) { fprintf(stderr, "key %lu not admitted\n", (unsigned long)k); _exit(5); }
                    printf("# pair %d %s=%d preload: owner objects=%zu (refilled %lu)\n", p, lab, (int)ro, cache.owner_objects(0),
                           (unsigned long)missing);
                    fflush(stdout);
                    S->owner_ready = 1;
                    while (!S->reader_done.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    cache.stop();
                    _exit(0);
                }
                for (int w = 0; !S->owner_ready.load(); w++) {
                    if (w > 300000) { fprintf(stderr, "reader: no owner after 600 s\n"); _exit(6); }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
                if (sched_setaffinity(0, sizeof cs, &cs)) { fprintf(stderr, "FATAL: cannot pin to cpu %d\n", cpu); _exit(4); }
                Client rd(&cache, 1);
                for (uint64_t k : keys) rd.get(k, buf.data(), &kind);   // install every local-index entry
                std::vector<uint32_t> cyc(gets), hc_cyc;
                std::vector<uint8_t> kd(gets);
                std::vector<uint64_t> pick(gets);
                uint64_t rng = seed + 1000003ULL * (uint64_t)p;
                Stats &rst = cache.stats(1);
                for (int rep = 0; rep <= reps; rep++) {
                    for (uint64_t i = 0; i < gets; i++) pick[i] = keys[sm64(rng) % nkeys];
                    rd.flush_stats();
                    const uint64_t mt0 = rst.mono_table, mb0 = rst.mono_backend, mo0 = rst.mono_ovf_backend, mm0 = rst.mono_bk_moved,
                                   mk0 = rst.mono_bk_kg;
                    double sum = 0;
                    for (uint64_t i = 0; i < gets; i++) {
                        uint64_t k = pick[i];
                        uint32_t req = (i & 15) == 0 ? ck.begin(k) : 0;
                        _mm_lfence();
                        uint64_t c0 = __rdtsc();
                        _mm_lfence();
                        uint32_t vlen = rd.get(k, buf.data(), &kind);
                        unsigned aux;
                        uint64_t c1 = __rdtscp(&aux);
                        _mm_lfence();
                        cyc[i] = (uint32_t)std::min<uint64_t>(c1 - c0, UINT32_MAX);
                        kd[i] = (uint8_t)kind;
                        sum += c1 - c0;
                        if ((i & 15) == 0) ck.verify(k, req, buf.data(), vlen, 1, kind);
                    }
                    uint64_t n[3] = {0, 0, 0};
                    hc_cyc.clear();
                    for (uint64_t i = 0; i < gets; i++) { n[kd[i]]++; if (kd[i] == FAST) hc_cyc.push_back(cyc[i]); }
                    std::sort(cyc.begin(), cyc.end());
                    std::sort(hc_cyc.begin(), hc_cyc.end());
                    auto q = [&](const std::vector<uint32_t> &v, double x) { return v.empty() ? 0.0 : v[(uint64_t)(x * (v.size() - 1) + 0.5)] / ghz; };
                    double hsum = 0;
                    for (uint32_t x : hc_cyc) hsum += x;
                    rd.flush_stats();
                    char mc[128] = "";
                    if (vm)
                        snprintf(mc, sizeof mc, ",%lu,%lu,%lu,%lu,%lu", (unsigned long)(rst.mono_table - mt0), (unsigned long)(rst.mono_backend - mb0),
                                 (unsigned long)(rst.mono_ovf_backend - mo0), (unsigned long)(rst.mono_bk_moved - mm0),
                                 (unsigned long)(rst.mono_bk_kg - mk0));
                    printf("CSV,%s,%d,%d,%d,%u,%lu,%lu,%lu,%lu,%lu,%lu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f%s\n", scheme_s.c_str(), (int)ro,
                           p, rep, vs, (unsigned long)hot, (unsigned long)nkeys, (unsigned long)gets, (unsigned long)n[0],
                           (unsigned long)n[1], (unsigned long)n[2], q(cyc, 0.5), q(cyc, 0.99), sum / gets / ghz, q(hc_cyc, 0.5),
                           q(hc_cyc, 0.10), q(hc_cyc, 0.90), q(hc_cyc, 0.99), hc_cyc.empty() ? 0.0 : hsum / hc_cyc.size() / ghz, mc);
                    fflush(stdout);
                }
                if (ck.errors()) {
                    for (auto &e : ck.first_errors()) printf("# error (%s=%d pair %d): %s\n", lab, (int)ro, p, e.c_str());
                    S->fail = 1;
                }
                fflush(stdout);
                S->reader_done = 1;
                cache.stop();
                _exit(0);
            }
            for (int left = 2; left > 0; left--) {   // if one host process ends badly, stop the other
                int st = 0;
                pid_t w = waitpid(-1, &st, 0);
                const int h = w == kid[0] ? 0 : 1;
                if (!WIFEXITED(st) || WEXITSTATUS(st)) {
                    printf("# host %d process of pair %d (%s=%d) ended badly: %s %d\n", h, p, lab, (int)ro,
                           WIFSIGNALED(st) ? "signal" : "exit", WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
                    bad++;
                    if (left == 2) { S->reader_done = 1; kill(kid[1 - h], SIGKILL); }
                }
            }
            if (S->fail.load()) bad++;
            printf("# pair %d %s=%d checker: gets_checked=%lu errors=%lu\n", p, lab, (int)ro, (unsigned long)S->cc.gets.load(),
                   (unsigned long)(S->cc.err_p1 + S->cc.err_p2 + S->cc.err_p3 + S->cc.err_bytes + S->cc.err_stale));
            fflush(stdout);
            Cache::shm_unlink(c.shm_name);
            munmap(sp, sim_bytes);
        }
    printf("# %s\n", bad ? "SOME PAIRS FAILED" : "all pairs ok");
    return bad != 0;
}
