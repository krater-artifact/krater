// Sample a measured RPC round trip (ns) for latency injection (replay --rpc-dist, the sharding baseline).
// Header-only, C and C++. Reads tables of measured round trips, one file per message kind, client and value size:
//   <dir>/<kind>_<client>_<vs>.txt for vs in 64, 256, 1024, 4096; each holds RPC_DIST_N integers
//   (ns, ascending; line i = quantile (i + 0.5) / N); '#' lines are skipped.
// Sampling: draw one uniform index i and return the value at i, interpolated linearly in value
// size between the two tables around vlen (same i in both, so the result is the quantile of an
// in-between size). vlen <= 64 uses the 64 B table; vlen > 4096 extrapolates with the per-quantile
// slope between the 1 KB and 4 KB tables (never below the 4 KB value).
//   rpc_dist d;  if (rpc_dist_load(&d, "<dir>", "rdma", "all")) abort();
//   uint64_t ns = rpc_dist_sample(&d, vlen, &rng_state);   // rng_state: any nonzero uint64_t
#ifndef RPC_DIST_H
#define RPC_DIST_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define RPC_DIST_N 10000
#define RPC_DIST_SIZES 4
static const uint32_t rpc_dist_vs[RPC_DIST_SIZES] = {64, 256, 1024, 4096};

typedef struct {
    uint32_t t[RPC_DIST_SIZES][RPC_DIST_N];
} rpc_dist;

static inline int rpc_dist_load(rpc_dist *d, const char *dir, const char *kind, const char *client) {
    for (int s = 0; s < RPC_DIST_SIZES; s++) {
        char path[4096], line[256];
        snprintf(path, sizeof path, "%s/%s_%s_%u.txt", dir, kind, client, rpc_dist_vs[s]);
        FILE *f = fopen(path, "r");
        if (!f) { perror(path); return -1; }
        int n = 0;
        while (fgets(line, sizeof line, f))
            if (line[0] != '#' && line[0] != '\n') {
                if (n == RPC_DIST_N) { n++; break; }
                d->t[s][n++] = (uint32_t)strtoul(line, 0, 10);
            }
        fclose(f);
        if (n != RPC_DIST_N) { fprintf(stderr, "%s: expected %d values\n", path, RPC_DIST_N); return -1; }
    }
    return 0;
}

static inline uint64_t rpc_dist_rng(uint64_t *s) {  // xorshift64*
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return *s * 0x2545F4914F6CDD1DULL;
}

static inline uint64_t rpc_dist_at(const rpc_dist *d, uint32_t i, uint32_t vlen) {
    if (vlen <= rpc_dist_vs[0]) return d->t[0][i];
    int s = 1;
    while (s < RPC_DIST_SIZES - 1 && vlen > rpc_dist_vs[s]) s++;  // tables s-1, s bracket vlen (or s = last)
    double lo = d->t[s - 1][i], hi = d->t[s][i];
    double x = ((double)vlen - rpc_dist_vs[s - 1]) / (rpc_dist_vs[s] - rpc_dist_vs[s - 1]);
    double v = lo + (hi - lo) * x;
    if (vlen > rpc_dist_vs[RPC_DIST_SIZES - 1] && v < hi) v = hi;
    return (uint64_t)(v + 0.5);
}

static inline uint64_t rpc_dist_sample(const rpc_dist *d, uint32_t vlen, uint64_t *rng) {
    return rpc_dist_at(d, (uint32_t)(rpc_dist_rng(rng) % RPC_DIST_N), vlen);
}
#endif
