// GET hit latency of the prototype with and without the lease check, on one reader host.
// Two hosts: host 0 owns every measured key (its shared area is bound to --data-node), host 1 is the reader;
// the reader thread is pinned to --reader-cpu and host 1's local structures are allocated on that CPU's node.
// Setup: every key gets the same value size (--vs); the owner admits them all (host-0 GETs), then the reader
// GETs every key once (slow path: installs its local-index entry). Measured: --gets uniformly random GETs on the
// reader, each timed with rdtsc (lfence-fenced), --reps reps after one warm-up rep; every 16th result is
// checked by the independent checker outside the timed region.
// Modes: --lease 0 (no lease check: lease=false, revoke=false) / --lease 1 --lease-us T. With a T much longer
// than the run, entries never expire and the difference is the check itself (epoch load + compare on the fast
// path, the epoch loads at both ends of a GET); with a short T, re-accesses farther apart than T go through
// the slow path (reported as fast / slow fractions).
// --clock-hit 1: the lease_clock_hit option (one TSC read per GET).
// Output: one "CSV,..." line per rep: lease,clock_hit,lease_ns,vs,hot_bytes,keys,rep,median_ns,p99_ns,mean_ns,
//   p10_ns,p90_ns,fast_frac,slow_frac,miss_frac,lease_expired,lease_retry
#include <dirent.h>
#include <getopt.h>
#include <numa.h>
#include <sched.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    fprintf(stderr, "usage: hitbench --vs B --hot-bytes N --data-node D --reader-cpu C --lease 0|1 [--lease-us 1000] [--clock-hit 0|1]\n"
                    "                [--gets 1000000] [--reps 5] [--seed 1]\n");
    exit(2);
}

int main(int argc, char **argv) {
    uint32_t vs = 64;
    uint64_t hot = 8 << 20, gets = 1000000, lease_us = 1000, seed = 1;
    int data_node = 0, cpu = 36, lease = 1, reps = 5, clock_hit = 0;
    static option opts[] = {{"vs", 1, 0, 'v'}, {"hot-bytes", 1, 0, 'h'}, {"data-node", 1, 0, 'd'}, {"reader-cpu", 1, 0, 'c'},
                            {"lease", 1, 0, 'l'}, {"lease-us", 1, 0, 'T'}, {"gets", 1, 0, 'g'}, {"reps", 1, 0, 'r'},
                            {"seed", 1, 0, 's'}, {"clock-hit", 1, 0, 'k'}, {0, 0, 0, 0}};
    int ch;
    while ((ch = getopt_long(argc, argv, "", opts, nullptr)) != -1) {
        switch (ch) {
        case 'v': vs = atoi(optarg); break;
        case 'h': hot = strtoull(optarg, 0, 0); break;
        case 'd': data_node = atoi(optarg); break;
        case 'c': cpu = atoi(optarg); break;
        case 'l': lease = atoi(optarg); break;
        case 'T': lease_us = strtoull(optarg, 0, 0); break;
        case 'g': gets = strtoull(optarg, 0, 0); break;
        case 'r': reps = atoi(optarg); break;
        case 's': seed = strtoull(optarg, 0, 0); break;
        case 'k': clock_hit = atoi(optarg); break;
        default: usage();
        }
    }
    if (vs < 16 || vs > MAX_VLEN) usage();
    int reader_node = numa_node_of_cpu(cpu);
    if (reader_node < 0) { fprintf(stderr, "FATAL: cpu %d offline?\n", cpu); return 4; }
    // The cache's service threads (handlers, executors, epoch threads) inherit the creating thread's affinity: give
    // them every CPU of the reader's and the data's nodes EXCEPT the reader CPU, and pin only the reader thread
    // (after start(); pinning before creating the cache would put all threads on the reader CPU).
    cpu_set_t svc; CPU_ZERO(&svc);
    for (int node : {reader_node, data_node}) {
        bitmask *bm = numa_allocate_cpumask();
        if (numa_node_to_cpus(node, bm) == 0)
            for (unsigned c = 0; c < bm->size; c++)
                if (numa_bitmask_isbitset(bm, c) && (int)c != cpu) CPU_SET(c, &svc);
        numa_free_cpumask(bm);
    }
    if (sched_setaffinity(0, sizeof svc, &svc)) { perror("sched_setaffinity (service cpus)"); return 4; }
    numa_set_preferred(reader_node);   // host-local structures (local index, ...) first-touched here
    uint32_t nl = obj_lines(vs);
    uint64_t obj_bytes = (uint64_t)nl * LINE, nkeys = hot / obj_bytes;
    // keys owned by host 0
    std::vector<uint64_t> keys;
    for (uint64_t k = 1; keys.size() < nkeys; k++)
        if (Cache::owner_of_static(k, 2) == 0) keys.push_back(k);
    uint64_t maxkey = keys.back();
    std::vector<uint32_t> sizes(maxkey + 1, vs);
    SimBackend be(maxkey, 0);
    be.set_sizes(sizes.data());
    Checker ck(maxkey, &be);
    Config cfg;
    cfg.hosts = 2;
    cfg.shm_node = data_node;
    cfg.heat_batch = 16;
    uint64_t ai = 1;
    while (ai < 2 * nkeys) ai <<= 1;
    cfg.ai_slots = ai;
    uint64_t li = 1;
    while (li < 2 * nkeys) li <<= 1;
    cfg.li_slots = li;
    // everything fits: slab = objects + 25%; budget charges objects + AI table + LI entries
    cfg.slab_bytes = ((nkeys * obj_bytes) * 5 / 4 + (64 << 20)) & ~(size_t)(LINE - 1);
    cfg.budget_bytes = cfg.slab_bytes + ai * 16 + nkeys * 2 * cfg.li_entry_bytes;
    cfg.ring_lines = 1 << 14;
    set_update(cfg.rules, "lease");   // this benchmark measures the lease check; --lease 0 = no check at all
    cfg.rules.lease = lease != 0;
    cfg.rules.lease_clock_hit = clock_hit != 0;
    cfg.rules.restart_wait = false;   // no restart here; with T = minutes it would hold the preload for T
    cfg.lease_ns = lease_us * 1000;
    cfg.on_publish = [&](uint64_t k, uint32_t v) { ck.on_publish(k, v); };
    double ghz = tsc_ghz();
    Cache cache(cfg, &be);
    printf("# hitbench: vs=%u lines=%u hot=%lu keys=%lu data_node=%d reader_cpu=%d (node %d) lease=%d lease_ns=%lu "
           "gets=%lu reps=%d tsc_ghz=%.4f\n# config: %s\n",
           vs, nl, (unsigned long)hot, (unsigned long)nkeys, data_node, cpu, reader_node, lease,
           (unsigned long)cfg.lease_ns, (unsigned long)gets, reps, ghz, cache.describe().c_str());
    fflush(stdout);
    cache.start();
    cache.wait_writable();
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    if (sched_setaffinity(0, sizeof cs, &cs)) { fprintf(stderr, "FATAL: cannot pin to cpu %d\n", cpu); return 4; }
    {   // record every thread's allowed CPUs (the reader must be alone on its CPU)
        std::string line = "# threads (tid:allowed cpus):";
        char path[64];
        DIR *d = opendir("/proc/self/task");
        while (dirent *de = d ? readdir(d) : nullptr) {
            if (de->d_name[0] == '.') continue;
            snprintf(path, sizeof path, "/proc/self/task/%s/status", de->d_name);
            if (FILE *f = fopen(path, "r")) {
                char l[512];
                while (fgets(l, sizeof l, f))
                    if (!strncmp(l, "Cpus_allowed_list:", 18)) {
                        l[strcspn(l, "\n")] = 0;
                        line += std::string(" ") + de->d_name + ":" + (l + 19 + strspn(l + 19, " \t"));
                    }
                fclose(f);
            }
        }
        if (d) closedir(d);
        printf("%s\n", line.c_str());
    }
    std::vector<char> buf(MAX_OUT);
    Kind kind;
    {   // owner admits every key (host-0 GETs: miss + local fill)
        Client o(&cache, 0);
        for (uint64_t k : keys) o.get(k, buf.data(), &kind);
        cache.wait_idle();
        uint64_t miss_again = 0;
        for (uint64_t k : keys)
            if (!cache.owner_cached(k)) { o.get(k, buf.data(), &kind); miss_again++; }
        cache.wait_idle();
        printf("# preload: owner objects=%zu (refilled %lu)\n", cache.owner_objects(0), (unsigned long)miss_again);
    }
    Client rd(&cache, 1);
    for (uint64_t k : keys) rd.get(k, buf.data(), &kind);   // install every local-index entry
    std::vector<uint32_t> cyc(gets);
    std::vector<uint64_t> pick(gets);
    uint64_t rng = seed;
    Stats &st = cache.stats(1);
    for (int rep = 0; rep <= reps; rep++) {
        for (uint64_t i = 0; i < gets; i++) pick[i] = keys[sm64(rng) % nkeys];
        rd.flush_stats();
        uint64_t f0 = st.get_fast, s0 = st.get_slow, m0 = st.get_miss, le0 = st.lease_expired, lr0 = st.lease_retry;
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
            sum += c1 - c0;
            if ((i & 15) == 0) ck.verify(k, req, buf.data(), vlen, 1, kind);
        }
        rd.flush_stats();
        uint64_t f = st.get_fast - f0, s = st.get_slow - s0, m = st.get_miss - m0;
        double tot = (double)(f + s + m);
        std::sort(cyc.begin(), cyc.end());
        auto q = [&](double p) { return cyc[(uint64_t)(p * (gets - 1) + 0.5)] / ghz; };
        printf("CSV,%d,%d,%lu,%u,%lu,%lu,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.4f,%.4f,%.4f,%lu,%lu\n", lease, clock_hit,
               (unsigned long)cfg.lease_ns, vs, (unsigned long)hot, (unsigned long)nkeys, rep, q(0.5), q(0.99),
               sum / gets / ghz, q(0.10), q(0.90), f / tot, s / tot, m / tot,
               (unsigned long)(st.lease_expired - le0), (unsigned long)(st.lease_retry - lr0));
        fflush(stdout);
    }
    cache.stop();
    printf("# checker: %s\n", ck.summary().c_str());
    for (auto &e : ck.first_errors()) printf("# error: %s\n", e.c_str());
    return ck.errors() != 0;
}
