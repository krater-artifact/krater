// Pointer-chasing latency probe (e.g. the load latency of a CXL node against local DRAM).
// Placement is controlled from outside: numactl --physcpubind=<cpu> --membind=<node> ./chase ...
// One cache line (64 B) per node, single random cycle (Sattolo), so every load depends on the previous one.
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <sched.h>

#define LINE 64

static uint64_t rng = 88172645463325252ULL;
static inline uint64_t xs(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static void print_numa_maps(void *addr) {
    char want[32], line[4096];
    snprintf(want, sizeof want, "%lx ", (unsigned long)addr);
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f) return;
    while (fgets(line, sizeof line, f))
        if (strncmp(line, want, strlen(want)) == 0) printf("# numa_maps: %s", line);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <bytes> <steps_per_rep> <reps> <thp 0|1> [seed]\n", argv[0]);
        return 1;
    }
    size_t bytes = strtoull(argv[1], 0, 0);
    uint64_t steps = strtoull(argv[2], 0, 0);
    int reps = atoi(argv[3]);
    int thp = atoi(argv[4]);
    if (argc > 5) rng = strtoull(argv[5], 0, 0) | 1;

    size_t n = bytes / LINE;
    char *buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(buf, bytes, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    memset(buf, 0, bytes);  // first touch under the numactl policy

    uint32_t *perm = malloc(n * sizeof *perm);
    for (size_t i = 0; i < n; i++) perm[i] = i;
    for (size_t i = n - 1; i > 0; i--) {  // Sattolo: one cycle covering all lines
        size_t j = xs() % i;
        uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (size_t i = 0; i < n; i++) *(void **)(buf + (size_t)i * LINE) = buf + (size_t)perm[i] * LINE;
    free(perm);

    print_numa_maps(buf);
    printf("# cpu=%d bytes=%zu lines=%zu steps=%lu thp=%d\n", sched_getcpu(), bytes, n, (unsigned long)steps, thp);

    void **p = (void **)buf;
    for (uint64_t i = 0; i < n; i++) p = *p;  // warm-up: one full lap (TLB, frequency ramp)

    for (int r = 0; r < reps; r++) {
        double t0 = now_ns();
        for (uint64_t i = 0; i < steps; i++) p = *p;
        double t1 = now_ns();
        printf("rep=%d ns_per_access=%.2f\n", r, (t1 - t0) / steps);
    }
    printf("# sink=%p cpu_end=%d\n", (void *)p, sched_getcpu());
    return 0;
}
