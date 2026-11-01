// H5 check: readers doing Krater's hit check while a single owner rewrites the objects in place.
//
// Shared memory (node C, e.g. the CXL node):
//   value array   n objects x value_size bytes, gen layout: line = [gen 4B][payload 60B]
//   auth index    key -> {gen, off}, 16 B per key, written only by the owner
// Owner thread (owner_cpu): at rate R (writes/s), picks a random key, allocates a fresh gen from a
//   single monotonically increasing 32-bit counter, rewrites the object IN PLACE line by line
//   (gen store first, then the 60 B payload; clwb after each full line), sfence, then publishes
//   {gen} in the auth index entry (clwb + sfence).
// Reader threads (reader cpus, other sockets): local index on their own node (key -> {gen, off}),
//   uniformly random keys, Krater's read (one aligned 64 B load per line, gen checked from the
//   register; all lines must carry the expected gen). check_mode 2/1/0 are controls (see below).
//   On a failed check (the slow path): count a detection, flush + fence + read the auth index entry,
//   flush the object's lines + fence, THEN install the gen in the local index and reread; repeat
//   until the check passes.
//   Every accepted read is verified independently: the payload must equal payload(key, gen, line).
//   A mismatch is an UNDETECTED error (must be 0).
// What it checks (H5): the owner stores each line's gen first, then its payload. A read that loads
// the line with ONE aligned 64 B load and finds the expected (old) gen must then hold the old
// payload, if that load reads one instance of the line. A load that tore -- old gen and part of the
// payload read before the owner's stores, the rest after -- would pass the gen check with a mixed
// payload, and the independent verification counts it as an UNDETECTED error. So 0 undetected
// errors under rewrites is evidence for H5 on this CPU. (check_mode 1, which reads the gen with a
// separate load before the payload, is the control: there such mixes do pass the check, and the
// verification counts them as undetected errors.)
//
// On a hardware-coherent machine the experiment exercises objects rewritten while being read (torn
// multi-line reads, a stale local index), not stale copies left in a CPU cache, which only exist on
// a non-coherent platform.
#define _GNU_SOURCE
#include <immintrin.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#define LINE 64
#define PAY 60
#define NREAD 3  // max readers; the run uses nread (1..NREAD) given by the reader CPU list
#define barrier() __asm__ volatile("" ::: "memory")

typedef struct { uint32_t gen, pad; uint64_t off; } AEnt;  // 16 B

static inline uint64_t sm64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static void *alloc_on(size_t bytes, int node) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    unsigned long mask = 1UL << node;
    if (mbind(p, bytes, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT)) { perror("mbind"); exit(1); }
    madvise(p, bytes, MADV_HUGEPAGE);
    memset(p, 0, bytes);
    return p;
}

static int verify_node(const char *tag, void *addr, size_t bytes, int want) {
    int ok = 0, bad = 0;
    for (int k = 0; k < 64; k++) {
        char *pg = (char *)addr + (bytes / 64) * k;
        int node = -1;
        if (get_mempolicy(&node, NULL, 0, pg, MPOL_F_NODE | MPOL_F_ADDR) == 0 && node == want) ok++; else bad++;
    }
    printf("# placement %s: want node%d ok=%d bad=%d\n", tag, want, ok, bad);
    return bad;
}

// Deterministic payload word w (0..6) of line l for (key, gen); bytes 56..59 use word 7's low 4 bytes.
static inline uint64_t pword(uint64_t key, uint32_t gen, uint64_t l, uint64_t w) {
    uint64_t z = key * 0x9E3779B97F4A7C15ULL ^ ((uint64_t)gen << 20) ^ (l * 8 + w) * 0xD6E8FEB86659FD93ULL;
    z ^= z >> 29; z *= 0xBF58476D1CE4E5B9ULL; z ^= z >> 32;
    return z;
}
static inline void fill_payload(char *dst, uint64_t key, uint32_t gen, uint64_t l) {
    uint64_t w[8];
    for (int i = 0; i < 8; i++) w[i] = pword(key, gen, l, i);
    memcpy(dst, w, PAY);
}

static uint64_t n, nlines;
static size_t vs;
static char *vals;        // CXL
static AEnt *auth;        // CXL
static volatile int stop, go;
static double rate;       // owner writes per second (0 = idle)
static uint32_t next_gen;
// check_mode 3 = Krater's check: one aligned 64 B load per line, gen checked from the register
//            2 = gen loaded before and after the payload, with separate loads (control)
//            1 = gen read once, before the payload (control: torn reads can slip through)
//            0 = no check at all                     (control)
static int check_mode = 3;
// store_order 0 = owner writes gen then payload in each line (protocol)
//             1 = payload then gen (counterexample: a reader can see new payload with the old gen)
static int store_order = 0;

typedef struct {
    int id, cpu, node;
    AEnt *lidx;
    char *buf;
    uint64_t rng;
    // results
    uint64_t ops, detections, retries, torn, stale, undetected;
    uint64_t retry_cycles;
    double secs;
    // per-detection slow-path cost samples (cycles), reservoir of at most RSAMP entries
    uint32_t *samp;
    uint64_t nsamp_seen;
} Reader;
#define RSAMP (1u << 20)

static void add_sample(Reader *r, uint64_t cyc) {
    uint32_t c = cyc > 0xffffffffULL ? 0xffffffffu : (uint32_t)cyc;
    if (r->nsamp_seen < RSAMP) r->samp[r->nsamp_seen] = c;
    else {
        uint64_t j = sm64(&r->rng) % (r->nsamp_seen + 1);
        if (j < RSAMP) r->samp[j] = c;
    }
    r->nsamp_seen++;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

// p in [0,1]; returns ns
static double pct_ns(uint32_t *v, uint64_t n, double p, double ghz) {
    if (n == 0) return 0;
    uint64_t i = (uint64_t)(p * (n - 1) + 0.5);
    return v[i] / ghz;
}

typedef struct { int cpu; uint64_t writes; double secs; uint64_t rng; } Owner;

static void pin(int cpu) {  // a failed pin must not be silent (e.g. CPU offline or outside our cpuset)
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    int e = pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    if (e) { fprintf(stderr, "FATAL: cannot pin to cpu %d (error %d): offline or outside the cpuset?\n", cpu, e); exit(4); }
}

static void write_object(uint64_t k, uint32_t g) {
    char *o = vals + (k - 1) * vs;
    char line[LINE] __attribute__((aligned(64)));
    for (uint64_t l = 0; l < nlines; l++) {
        volatile uint32_t *gp = (volatile uint32_t *)(o + l * LINE);
        fill_payload(line, k, g, l);
        if (store_order == 0) {
            *gp = g;                               // gen first
            barrier();
            memcpy(o + l * LINE + 4, line, PAY);   // then payload
        } else {                                   // counterexample: payload first, gen last
            memcpy(o + l * LINE + 4, line, PAY);
            barrier();
            *gp = g;
        }
        barrier();
        _mm_clwb(o + l * LINE);                // full line written -> write back
    }
    _mm_sfence();
    AEnt *a = &auth[k];
    ((volatile AEnt *)a)->gen = g;
    _mm_clwb(a);
    _mm_sfence();
}

static void *owner_main(void *arg) {
    Owner *ow = arg;
    pin(ow->cpu);
    while (!go) ;
    double t0 = now_ns();
    uint64_t w = 0;
    while (!stop) {
        if (rate <= 0) { usleep(1000); continue; }
        double due = t0 + w * 1e9 / rate;
        while (now_ns() < due && !stop) _mm_pause();
        if (stop) break;
        uint64_t k = (sm64(&ow->rng) % n) + 1;
        write_object(k, next_gen++);
        w++;
    }
    ow->writes = w;
    ow->secs = (now_ns() - t0) / 1e9;
    return NULL;
}

// One read attempt. Returns 0 if all lines passed; 1 = stale (consistent but != expected), 2 = torn.
static inline int try_read(Reader *r, uint64_t k, uint32_t g) {
    const char *o = vals + (k - 1) * vs;
    int res = 0;
    if (check_mode == 3) {  // one 64 B load per line; lines with different gens = torn object
        uint32_t g0 = 0;
        for (uint64_t l = 0; l < nlines; l++) {
            __m512i x = _mm512_load_si512((const void *)(o + l * LINE));
            __asm__ volatile("" : "+v"(x));  // one load per line, gen taken from the register
            uint32_t gl = (uint32_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(x));
            _mm512_mask_storeu_epi8(r->buf + l * PAY - 4, (__mmask64)0xFFFFFFFFFFFFFFF0ULL, x);
            if (l == 0) g0 = gl;
            else if (gl != g0) res = 2;
        }
        if (res == 0 && g0 != g) res = 1;
        return res;
    }
    for (uint64_t l = 0; l < nlines; l++) {
        const volatile uint32_t *gp = (const volatile uint32_t *)(o + l * LINE);
        uint32_t g1 = *gp;
        barrier();
        memcpy(r->buf + l * PAY, o + l * LINE + 4, PAY);
        barrier();
        uint32_t g2 = *gp;
        if (check_mode == 2 && g1 != g2) res = 2;
        else if (check_mode >= 1 && g1 != g && res == 0) res = 1;
    }
    return res;
}

static void *reader_main(void *arg) {
    Reader *r = arg;
    pin(r->cpu);
    while (!go) ;
    double t0 = now_ns();
    uint64_t ops = 0;
    char expect[PAY] __attribute__((aligned(64)));
    while (!stop) {
        for (int b = 0; b < 64; b++) {
            uint64_t k = (sm64(&r->rng) % n) + 1;
            uint32_t g = r->lidx[k].gen;
            int res = try_read(r, k, g);
            if (res) {
                r->detections++;
                if (res == 2) r->torn++; else r->stale++;
                uint64_t c0 = __rdtsc();
                // Slow path, in Krater's order: flush +
                // fence + read the authoritative index entry; flush the object's lines + fence;
                // only THEN install the gen in the local index (other reader threads of this host
                // must not see the new gen before the stale lines are gone); reread.
                while (res) {
                    const char *o = vals + (k - 1) * vs;
                    _mm_clflushopt(&auth[k]);
                    _mm_mfence();
                    g = ((volatile AEnt *)&auth[k])->gen;
                    // The object address o comes from k, not from the entry just read, and
                    // clflushopt is not ordered with earlier loads to other lines, so without this
                    // fence the object flushes could run before the index read (the model's
                    // flush-index-install counterexample order). The fence keeps the design's order:
                    // read the index, THEN flush the object's lines.
                    _mm_mfence();
                    for (uint64_t l = 0; l < nlines; l++) _mm_clflushopt((void *)(o + l * LINE));
                    _mm_mfence();
                    r->lidx[k].gen = g;
                    res = try_read(r, k, g);
                    r->retries++;
                }
                uint64_t dc = __rdtsc() - c0;
                r->retry_cycles += dc;
                add_sample(r, dc);
            }
            // independent verification of the accepted read
            for (uint64_t l = 0; l < nlines; l++) {
                fill_payload(expect, k, g, l);
                if (memcmp(expect, r->buf + l * PAY, PAY)) { r->undetected++; break; }
            }
            ops++;
        }
    }
    r->secs = (now_ns() - t0) / 1e9;
    r->ops = ops;
    return NULL;
}

static double tsc_ghz(void) {
    double t0 = now_ns(); uint64_t c0 = __rdtsc();
    usleep(200000);
    return (__rdtsc() - c0) / (now_ns() - t0);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <value_size> <hotset_bytes> <rate_per_s> <secs> <rep> <cxl_node> [check_mode 3|2|1|0] [store_order 0|1] [owner_cpu] [reader cpus,comma]\n", argv[0]);
        return 1;
    }
    vs = strtoull(argv[1], 0, 0);
    uint64_t hot = strtoull(argv[2], 0, 0);
    rate = atof(argv[3]);
    double secs = atof(argv[4]);
    int rep = atoi(argv[5]), cn = atoi(argv[6]);
    if (argc > 7) check_mode = atoi(argv[7]);
    if (argc > 8) store_order = atoi(argv[8]);
    int owner_cpu = argc > 9 ? atoi(argv[9]) : 36;
    int rcpu[NREAD] = {4, 68, 100}, rnode[NREAD], nread = NREAD;
    if (argc > 10) {
        int j = 0;
        char *t = strtok(argv[10], ",");
        for (; t && j < NREAD; t = strtok(NULL, ",")) rcpu[j++] = atoi(t);
        if (j < 1 || t) { fprintf(stderr, "need 1..%d reader cpus\n", NREAD); return 2; }
        nread = j;
    }
    for (int j = 0; j < nread; j++) {
        rnode[j] = numa_node_of_cpu(rcpu[j]);
        if (rnode[j] < 0) { fprintf(stderr, "FATAL: cpu %d has no node (offline?)\n", rcpu[j]); return 4; }
    }
    n = hot / vs;
    nlines = vs / LINE;

    vals = alloc_on(hot, cn);
    auth = alloc_on((n + 1) * sizeof(AEnt), cn);
    for (uint64_t k = 1; k <= n; k++) {
        uint32_t g = (uint32_t)k;  // initial gens 1..n; the owner continues from n+1
        char *o = vals + (k - 1) * vs;
        for (uint64_t l = 0; l < nlines; l++) {
            *(uint32_t *)(o + l * LINE) = g;
            fill_payload(o + l * LINE + 4, k, g, l);
        }
        auth[k] = (AEnt){g, 0, (k - 1) * vs};
    }
    next_gen = (uint32_t)n + 1;

    const int *RCPU = rcpu, *RNODE = rnode;
    Reader rd[NREAD];
    int bad = verify_node("cxl.val", vals, hot, cn) + verify_node("cxl.auth", auth, (n + 1) * sizeof(AEnt), cn);
    for (int i = 0; i < nread; i++) {
        memset(&rd[i], 0, sizeof rd[i]);
        rd[i].id = i; rd[i].cpu = RCPU[i]; rd[i].node = RNODE[i];
        rd[i].rng = 0xABCDEFULL * (i + 1) + rep;
        rd[i].lidx = alloc_on((n + 1) * sizeof(AEnt), RNODE[i]);
        rd[i].samp = alloc_on(RSAMP * sizeof(uint32_t), RNODE[i]);
        memcpy(rd[i].lidx, auth, (n + 1) * sizeof(AEnt));
        rd[i].buf = (char *)aligned_alloc(LINE, nlines * PAY + 2 * LINE) + LINE;  // headroom for the masked store
        char tag[32]; snprintf(tag, sizeof tag, "r%d.lidx", i);
        bad += verify_node(tag, rd[i].lidx, (n + 1) * sizeof(AEnt), RNODE[i]);
    }
    if (bad) { fprintf(stderr, "placement check failed\n"); return 2; }
    Owner ow = {owner_cpu, 0, 0, 0x5151ULL + rep};
    double ghz = tsc_ghz();
    printf("# vs=%zu hot=%lu n=%lu rate=%.0f secs=%.1f rep=%d tsc_ghz=%.3f owner_cpu=%d reader_cpus=%d,%d,%d check_mode=%d store_order=%d nread=%d\n",
           vs, (unsigned long)hot, (unsigned long)n, rate, secs, rep, ghz, owner_cpu, rcpu[0], nread > 1 ? rcpu[1] : -1,
           nread > 2 ? rcpu[2] : -1, check_mode, store_order, nread);

    pthread_t to, tr[NREAD];
    pthread_create(&to, NULL, owner_main, &ow);
    for (int i = 0; i < nread; i++) pthread_create(&tr[i], NULL, reader_main, &rd[i]);
    usleep(100000);
    go = 1;
    usleep((useconds_t)(secs * 1e6));
    stop = 1;
    pthread_join(to, NULL);
    for (int i = 0; i < nread; i++) pthread_join(tr[i], NULL);

    uint64_t undetected = 0;
    printf("# owner writes=%lu achieved_rate=%.0f\n", (unsigned long)ow.writes, ow.writes / ow.secs);
    for (int i = 0; i < nread; i++) {
        Reader *r = &rd[i];
        double retry_ns = r->detections ? r->retry_cycles / ghz / r->detections : 0;
        uint64_t ns_ = r->nsamp_seen < RSAMP ? r->nsamp_seen : RSAMP;
        qsort(r->samp, ns_, sizeof(uint32_t), cmp_u32);
        double p50 = pct_ns(r->samp, ns_, 0.50, ghz), p90 = pct_ns(r->samp, ns_, 0.90, ghz),
               p99 = pct_ns(r->samp, ns_, 0.99, ghz), pmax = ns_ ? r->samp[ns_ - 1] / ghz : 0;
        printf("CSV,%d,%d,%zu,%lu,%.0f,%.0f,%d,%d,%d,%.3f,%lu,%lu,%lu,%lu,%lu,%.1f,%lu,%.0f,%.1f,%.1f,%.1f,%.1f,%lu\n", check_mode, store_order, vs, (unsigned long)hot, rate,
               ow.writes / ow.secs, rep, i, r->cpu, r->ops / r->secs / 1e6, (unsigned long)r->ops,
               (unsigned long)r->detections, (unsigned long)r->stale, (unsigned long)r->torn, (unsigned long)r->retries,
               retry_ns, (unsigned long)r->undetected, r->secs * 1e9 / r->ops, p50, p90, p99, pmax, (unsigned long)ns_);
        undetected += r->undetected;
    }
    printf("# undetected_total=%lu\n", (unsigned long)undetected);
    return (undetected != 0 && check_mode == 3 && store_order == 0) ? 3 : 0;
}
