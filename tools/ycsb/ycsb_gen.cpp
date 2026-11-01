// YCSB-B / YCSB-C request traces in the binary record format of trace_io.h (records.bin + .meta + objects.bin), for
// replay --bin.
//   ycsb_gen [--workload=B|C] [--sizes=ycsb|cluster17] [--keys=10000000] [--requests=100000000] [--clients=1000]
//            [--rate=36630] [--seed=1] [--c17=<objects.csv>] [--folded=1] [--theta=0.99] <out_dir>
// Request keys: YCSB's ScrambledZipfianGenerator (core/src/main/java/site/ycsb/generator/ScrambledZipfianGenerator.java):
// a ZipfianGenerator over ITEM_COUNT = 10^10 items with theta 0.99 and its precomputed zeta (26.46902820178302), then
// fnvhash64(rank) % keys (YCSB's FNV-1 64-bit over the 8 bytes of the rank, low byte first, then Math.abs). This is what
// YCSB uses for requestdistribution=zipfian (workloads B and C).
// --folded=0: the Zipf is drawn on the key count itself (no folding of 10^10 ranks): rank = Zipf over
// --keys items with exponent --theta, key = fnvhash64(rank) % keys (the YCSB-C++ style). For theta < 1 the rank comes from
// YCSB's ZipfianGenerator algorithm (Gray et al.) with zeta(keys, theta) computed exactly; for theta >= 1 that algorithm is
// not defined, so the rank is drawn by exact inverse CDF (binary search over the cumulative 1 / i^theta).
// Ops: B = 95% read (GET, op 2) / 5% update (write, op 1); C = 100% read. No load phase: the cache starts empty and the
// backend holds every key (the replay's miss path fills it).
// Clients: each request's client is uniform over --clients ids "client<i>"; the record keeps the top 8 bits of
// KeyTab::hash("client<i>"), so --routing client sends each client's requests to one host, as for the Twitter traces.
// Timestamps: request i at second floor(i / rate) (default 36630 /s = cluster17's 100 M-request window, 2730 s), for
// --pace trace.
// Object sizes (objects.bin, u32 per key id, index 0 unused; ids in order of first appearance):
//   ycsb      = 24 B key ("user" + 20 digits) + 1000 B value (YCSB's default 10 fields x 100 B) = 1024 B
//   cluster17 = each key's size drawn (seeded by the key) from the object sizes of cluster17's window (objects.csv).
#include "trace_io.h"
#include <cmath>
#include <random>
#include <string>
#include <sys/stat.h>

struct Zipf {  // site.ycsb.generator.ZipfianGenerator(0, items - 1, theta, zetan)
  double items, theta, zetan, alpha, eta, zeta2;
  Zipf(double n, double th, double zn) : items(n), theta(th), zetan(zn) {
    zeta2 = 1.0 + 1.0 / std::pow(2.0, theta); alpha = 1.0 / (1.0 - theta);
    eta = (1.0 - std::pow(2.0 / items, 1.0 - theta)) / (1.0 - zeta2 / zetan);
  }
  uint64_t next(double u) const {
    double uz = u * zetan;
    if (uz < 1.0) return 0;
    if (uz < 1.0 + std::pow(0.5, theta)) return 1;
    return (uint64_t)(items * std::pow(eta * u - eta + 1.0, alpha));
  }
};
static uint64_t fnvhash64(uint64_t val) {  // site.ycsb.Utils.fnvhash64
  int64_t h = (int64_t)0xCBF29CE484222325ull;
  for (int i = 0; i < 8; i++) { int64_t octet = (int64_t)(val & 0xff); val >>= 8; h ^= octet; h = (int64_t)((uint64_t)h * 1099511628211ull); }
  return (uint64_t)(h < 0 ? -h : h);  // Math.abs (Long.MIN_VALUE stays negative in Java; vanishingly rare, kept positive here)
}

int main(int argc, char **argv) {
  Opts opt(argc, argv);
  if (argc < 2) { fprintf(stderr, "usage: %s [--workload=B|C] [--sizes=ycsb|cluster17] [--keys=N] [--requests=N] [--clients=N] [--rate=R] [--seed=S] [--c17=objects.csv] out_dir\n", argv[0]); return 2; }
  std::string out = argv[1], wl = opt.get("workload", "B"), sizes = opt.get("sizes", "ycsb");
  uint64_t nkeys = strtoull(opt.get("keys", "10000000").c_str(), nullptr, 10), nreq = strtoull(opt.get("requests", "100000000").c_str(), nullptr, 10);
  uint64_t nclients = strtoull(opt.get("clients", "1000").c_str(), nullptr, 10), seed = strtoull(opt.get("seed", "1").c_str(), nullptr, 10);
  double rate = atof(opt.get("rate", "36630").c_str()), readp = wl == "C" ? 1.0 : 0.95;
  bool folded = opt.get("folded", "1") != "0"; double theta = atof(opt.get("theta", "0.99").c_str());
  if (wl != "B" && wl != "C") { fprintf(stderr, "workload B or C\n"); return 2; }
  std::vector<uint32_t> c17;
  if (sizes == "cluster17") {
    std::string f = opt.get("c17"); if (f.empty()) { fprintf(stderr, "--sizes=cluster17 needs --c17=<objects.csv>\n"); return 2; }
    FILE *fp = fopen(f.c_str(), "r"); if (!fp) { perror(f.c_str()); return 1; }
    char line[512]; if (!fgets(line, sizeof line, fp)) return 1;  // wid,full_id,size,reqs,gets
    while (fgets(line, sizeof line, fp)) { unsigned a, b, s; if (sscanf(line, "%u,%u,%u", &a, &b, &s) == 3) c17.push_back(s); }
    fclose(fp); fprintf(stderr, "cluster17 sizes: %zu objects\n", c17.size());
  } else if (sizes != "ycsb") { fprintf(stderr, "sizes ycsb or cluster17\n"); return 2; }
  mkdir(out.c_str(), 0775);
  std::vector<uint64_t> chash(nclients);
  for (uint64_t c = 0; c < nclients; c++) { std::string s = "client" + std::to_string(c); chash[c] = KeyTab::hash(s); }
  Zipf z(1e10, 0.99, 26.46902820178302);
  std::vector<double> cdf;  // unfolded, theta >= 1
  double zn = 0;
  if (!folded) {
    if (theta < 1) { for (uint64_t i = 1; i <= nkeys; i++) zn += std::pow((double)i, -theta); z = Zipf((double)nkeys, theta, zn); }
    else { cdf.resize(nkeys); double c = 0; for (uint64_t i = 0; i < nkeys; i++) { c += std::pow((double)(i + 1), -theta); cdf[i] = c; } for (auto &x : cdf) x /= c; }
    fprintf(stderr, "unfolded Zipf: theta %g over %lu keys (%s)\n", theta, (unsigned long)nkeys, theta < 1 ? "YCSB ZipfianGenerator" : "exact inverse CDF");
  }
  std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(0.0, 1.0);
  std::vector<uint32_t> id(nkeys, 0); std::vector<uint64_t> key_of{0};  // id -> key index
  FILE *fo = fopen((out + "/records.bin").c_str(), "wb"); if (!fo) { perror("records.bin"); return 1; }
  std::vector<uint64_t> buf; buf.reserve(1 << 20);
  uint64_t gets = 0, writes = 0, max_ts = 0;
  for (uint64_t i = 0; i < nreq; i++) {
    uint64_t rank;
    if (folded || theta < 1) { rank = z.next(U(rng)); if (!folded && rank >= nkeys) rank = nkeys - 1; }
    else { double u = U(rng); rank = (uint64_t)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin()); if (rank >= nkeys) rank = nkeys - 1; }
    uint64_t k = fnvhash64(rank) % nkeys;
    if (!id[k]) { key_of.push_back(k); id[k] = (uint32_t)(key_of.size() - 1); }
    int op = U(rng) < readp ? 2 : 1; (op == 2 ? gets : writes)++;
    uint64_t c = rng() % nclients; uint32_t ts = (uint32_t)((double)i / rate);
    if (ts > REC_TS_MAX) { fprintf(stderr, "timestamp overflow\n"); return 1; }
    max_ts = ts; buf.push_back(pack_rec(id[k], op, chash[c], ts));
    if (buf.size() == buf.capacity()) { fwrite(buf.data(), 8, buf.size(), fo); buf.clear(); }
  }
  fwrite(buf.data(), 8, buf.size(), fo); fclose(fo);
  uint64_t nd = key_of.size() - 1;
  std::vector<uint32_t> sz(nd + 1, 0); uint64_t bytes = 0;
  for (uint64_t j = 1; j <= nd; j++) {
    if (sizes == "ycsb") sz[j] = 1024;
    else { std::mt19937_64 r(key_of[j] * 0x9E3779B97F4A7C15ull ^ seed); sz[j] = c17[r() % c17.size()]; }
    bytes += sz[j];
  }
  FILE *fz = fopen((out + "/objects.bin").c_str(), "wb"); fwrite(sz.data(), 4, sz.size(), fz); fclose(fz);
  FILE *fm = fopen((out + "/records.bin.meta").c_str(), "w");
  fprintf(fm, "format u64:key32,op3,client_hash_top8,ts21\nlines %lu\nbad_lines 0\nrecords %lu\ndistinct_keys %lu\nclients %lu\nmax_ts %lu\n"
              "op_write %lu\nop_get %lu\nop_incr 0\nop_delete 0\nworkload ycsb-%s\nsizes %s\nkey_space %lu\ntheta %g\nfolded %d\nseed %lu\nrate %g\ndistinct_bytes %lu\n",
          (unsigned long)nreq, (unsigned long)nreq, (unsigned long)nd, (unsigned long)nclients, (unsigned long)max_ts, (unsigned long)writes,
          (unsigned long)gets, wl.c_str(), sizes.c_str(), (unsigned long)nkeys, folded ? 0.99 : theta, (int)folded, (unsigned long)seed, rate, (unsigned long)bytes);
  fclose(fm);
  fprintf(stderr, "ycsb-%s %s: %lu requests (%lu GET, %lu update), %lu distinct keys of %lu, %lu bytes, max_ts %lu\n", wl.c_str(), sizes.c_str(),
          (unsigned long)nreq, (unsigned long)gets, (unsigned long)writes, (unsigned long)nd, (unsigned long)nkeys, (unsigned long)bytes, (unsigned long)max_ts);
  return 0;
}
