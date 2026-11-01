// Does a read-only mapping change the cost of the reader's accesses? The SAME shm pages mapped twice in one process
// -- one view PROT_READ (as ro_peers maps another owner's block), one PROT_READ | PROT_WRITE (the default mapping) -- and
// the reader's access kinds timed in blocks that alternate between the views (same physical pages, same process and CPU: only
// the PTEs' permission differs). Both views reach the same lines, so before every timed run an untimed pass over the same lines
// THROUGH THE SAME VIEW sets their state (otherwise whichever view runs second finds the lines the first one just warmed).
// Kinds, each on uniformly random 64 B lines of the region:
//   load        one 64 B line load (two 32 B loads without AVX-512F); the pass loads the lines (cached, TLB warm)
//   coldload    the same load; the pass flushes the lines (the load comes from memory)
//   flushload   clflushopt + mfence + load (the slow path's / VER's AI-entry read, XTRA's directory word); the pass loads them
//   flush       clflushopt + mfence (a line the reader flushes and does not read right away); the pass loads them
// Every op is timed alone: lfence; rdtsc ... rdtscp; lfence. The region is bound to --node (mbind), the thread pinned to --cpu.
// Output: CSV,kind,view,block,bytes,ops,median_ns,mean_ns,p99_ns (view ro / rw); --blocks blocks per view and kind, the order of
// the views swapped every block.
#include <fcntl.h>
#include <getopt.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

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
    return (c1 - c0) / std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
}
static inline void load_line(const char *p) {
#if defined(__AVX512F__)
    __m512i x;
    __asm__ volatile("vmovdqa64 %1, %0" : "=v"(x) : "m"(*(const char(*)[64])p));
#else
    __m256i a, b;
    __asm__ volatile("vmovdqa %1, %0" : "=x"(a) : "m"(*(const char(*)[32])p));
    __asm__ volatile("vmovdqa %1, %0" : "=x"(b) : "m"(*(const char(*)[32])(p + 32)));
#endif
}

int main(int argc, char **argv) {
    uint64_t bytes = 16 << 20, ops = 200000, seed = 1;
    int node = 3, cpu = 52, blocks = 10;
    static option opts[] = {{"bytes", 1, 0, 'b'}, {"node", 1, 0, 'n'}, {"cpu", 1, 0, 'c'}, {"ops", 1, 0, 'o'}, {"blocks", 1, 0, 'k'},
                            {"seed", 1, 0, 's'}, {0, 0, 0, 0}};
    int ch;
    while ((ch = getopt_long(argc, argv, "", opts, nullptr)) != -1) {
        switch (ch) {
        case 'b': bytes = strtoull(optarg, 0, 0); break;
        case 'n': node = atoi(optarg); break;
        case 'c': cpu = atoi(optarg); break;
        case 'o': ops = strtoull(optarg, 0, 0); break;
        case 'k': blocks = atoi(optarg); break;
        case 's': seed = strtoull(optarg, 0, 0); break;
        default: fprintf(stderr, "usage: roflush [--bytes N] [--node D] [--cpu C] [--ops N] [--blocks K] [--seed S]\n"); return 2;
        }
    }
    bytes = (bytes + 4095) & ~4095ULL;
    std::string name = "/rcroflush-" + std::to_string(getpid());
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0 || ftruncate(fd, (off_t)bytes)) { perror("shm"); return 1; }
    char *rw = (char *)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (rw == MAP_FAILED) { perror("mmap rw"); return 1; }
    unsigned long mask = 1UL << node;
    if (mbind(rw, bytes, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT)) { perror("mbind"); return 1; }
    for (uint64_t i = 0; i < bytes; i += 64) memset(rw + i, (int)(i >> 6), 64);
    char *ro = (char *)mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (ro == MAP_FAILED) { perror("mmap ro"); return 1; }
    close(fd);
    shm_unlink(name.c_str());   // the mappings keep it
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    if (sched_setaffinity(0, sizeof cs, &cs)) { perror("sched_setaffinity"); return 4; }
    const double ghz = tsc_ghz();
    std::vector<uint64_t> off(ops);
    uint64_t rng = seed;
    for (auto &o : off) o = (sm64(rng) % (bytes / 64)) * 64;
    for (uint64_t i = 0; i < ops; i++) { (void)*(volatile char *)(ro + off[i]); (void)*(volatile char *)(rw + off[i]); }   // map both views' PTEs
    printf("# roflush: bytes=%lu node=%d cpu=%d ops=%lu blocks=%d tsc_ghz=%.4f%s\n", (unsigned long)bytes, node, cpu,
           (unsigned long)ops, blocks, ghz,
#if defined(__AVX512F__)
           ""
#else
           " (NO AVX-512F: a line load is two 32 B loads)"
#endif
    );
    printf("CSV_HEADER,kind,view,block,bytes,ops,median_ns,mean_ns,p99_ns\n");
    static const char *KN[] = {"load", "coldload", "flushload", "flush"};
    std::vector<uint32_t> cyc(ops);
    for (int b = 0; b < blocks; b++)
        for (int kind = 0; kind < 4; kind++)
            for (int j = 0; j < 2; j++) {
                const bool use_ro = (b % 2 == 0) ? j == 0 : j == 1;
                const char *base = use_ro ? ro : rw;
                for (uint64_t i = 0; i < ops; i++) {   // untimed pass through this view: the lines' state
                    const char *p = base + off[(i + (uint64_t)b * 7919) % ops];
                    if (kind == 1) _mm_clflushopt((void *)p);
                    else load_line(p);
                }
                _mm_mfence();
                double sum = 0;
                for (uint64_t i = 0; i < ops; i++) {
                    const char *p = base + off[(i + (uint64_t)b * 7919) % ops];
                    _mm_lfence();
                    uint64_t c0 = __rdtsc();
                    _mm_lfence();
                    if (kind <= 1) load_line(p);
                    else {
                        _mm_clflushopt((void *)p);
                        _mm_mfence();
                        if (kind == 2) load_line(p);
                    }
                    unsigned aux;
                    uint64_t c1 = __rdtscp(&aux);
                    _mm_lfence();
                    cyc[i] = (uint32_t)std::min<uint64_t>(c1 - c0, UINT32_MAX);
                    sum += c1 - c0;
                }
                std::sort(cyc.begin(), cyc.end());
                printf("CSV,%s,%s,%d,%lu,%lu,%.1f,%.1f,%.1f\n", KN[kind], use_ro ? "ro" : "rw", b, (unsigned long)bytes,
                       (unsigned long)ops, cyc[ops / 2] / ghz, sum / ops / ghz, cyc[(uint64_t)(0.99 * (ops - 1))] / ghz);
                fflush(stdout);
            }
    return 0;
}
