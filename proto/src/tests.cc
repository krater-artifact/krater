// Fixed-point tests: targeted interleavings, forced with blocking hooks (gates), each the counterexample of one protocol
// rule. Each scenario must produce a checker error when its rule is removed (the variants in its must_fail_in list) and
// none with all rules on (the suite's must-pass variants). Plus a few basic path tests. Suites (--suite): revoke (the
// invalidation protocol: tombstone, bump2, pending, install, deadline, slowput, privwriter), lease, log, feed (the
// default protocol: log + change feed + kg in CXL), xtra (XTRA-style own freshness), delete (DELETE / EXPIRE in every
// scheme), ro (read-only peer blocks), mono (monotonic reads). Exit status 0 iff every expectation holds.
//   tests [-v]          threads mode: all hosts are threads of this process
//   tests [-v] --procs  procs mode: one child process per host on a POSIX shm area; the scenario code is the same, its
//                       operations become commands to the children, and the hook gates live in shared memory.
// Scenarios that need stale cache lines (install, the generation-wrap scenarios W1-W4, F4) run with the software
// non-coherent cache emulation.
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <deque>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <thread>

#include "rackcache.h"
#include "sim.h"

using namespace rc;

static bool g_procs = false;
static uint64_t g_put_delay_ns = 0;   // backend write latency of the next Env (A1-asyncput); 0 = none
static int g_put_key_after = -1;    // --put-key-after reply|publish (default: Config's)
static int g_put_async = -1;        // --put-async 0|1 (default: Config's)
static int g_obj_read = -1;         // --obj-read copy|prefetch|mstore|both (default: Config's)
static int g_hit_clock = 1;         // --hit-clock lfence3|rdtscp (rdtscp is the default)
static bool g_force_li32 = false;  // --li32: 32 B local-index entries (16 B entries are the default)
static int g_li16_fp_bits = 20;    // ... with this many fingerprint bits (small = forced aliasing)
static int g_li16_ep_bits = 16;    // ... and drop-generation bits (small = the wrap clear runs every few drop-alls)
static bool g_force_emu = false;   // --emu: every scenario runs with the software non-coherent reader caches
static bool g_ro_peers = false;    // --procs --ro-peers 1: every child maps the other host's block read-only
static bool g_mono = false;        // --mono 1: Config::mono (monotonic reads) in every scenario whose configuration allows it
static bool g_check_mono = false;  // P6 (monotonic reads) is counted as an error (the mono suite)
static int g_xrep = 0;             // XREP (--xtra-dir-replicas 1|2): a directory copy per host wherever XTRA own freshness runs
// the configurations Config::mono runs in (Cache::Impl::check_cfg refuses the others)
static bool mono_ok(const Config &c) {
    return c.rules.log && !c.rules.lease && !c.rules.revoke && c.rules.kg_cxl && c.scheme != Scheme::PRIV && c.scheme != Scheme::SHARD &&
           c.scheme != Scheme::VER && c.scheme != Scheme::VER_NOFLUSH && !(c.scheme == Scheme::XTRA && c.xtra_own_freshness);
}

static bool wait_until(const std::function<bool()> &f, int ms = 5000) {
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (!f()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

// ---------------------------------------------------------------- gates (process-shared in --procs mode)
struct Gate {
    std::atomic<int> used{0}, armed{0}, reached{0}, released{0}, timed_out{0};
    int pt = 0, host = 0;
    uint64_t key = 0;
};
constexpr int NGATE = 16;
static Gate g_local_gates[NGATE];
static Gate *g_gates = g_local_gates;   // points into the shared segment in --procs mode

static void gate_hook(Hook pt, int host, uint64_t key) {
    Gate *hit = nullptr;
    for (int i = 0; i < NGATE && !hit; i++) {
        Gate &x = g_gates[i];
        int one = 1;
        if (x.used && x.pt == (int)pt && x.host == host && x.key == key && x.armed.compare_exchange_strong(one, 0)) hit = &x;
    }
    if (!hit) return;
    hit->reached = 1;
    if (!wait_until([&] { return hit->released.load() != 0; }, 10000)) hit->timed_out = 1;
}
static Gate &new_gate(Hook pt, int host, uint64_t key) {
    for (int i = 0; i < NGATE; i++) {
        Gate &x = g_gates[i];
        if (x.used) continue;
        x.armed = 0; x.reached = 0; x.released = 0; x.timed_out = 0;
        x.pt = (int)pt; x.host = host; x.key = key;
        x.used = 1;
        return x;
    }
    abort();
}
static void arm(Gate &g) { g.armed = 1; }
static void disarm_all() {
    for (int i = 0; i < NGATE; i++) { g_gates[i].armed = 0; g_gates[i].released = 1; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for (int i = 0; i < NGATE; i++) g_gates[i].used = 0;
}

// ---------------------------------------------------------------- environment: local (threads) or one process per host (procs)
constexpr uint64_t NKEYS = 1024;
constexpr int NSLOT = 4;   // concurrent commands per child
enum Op { OP_GET = 1, OP_PUT, OP_EVICT, OP_CACHED, OP_OFFSET, OP_PREFETCH, OP_DROP, OP_PAUSE_EPOCH, OP_UNREACH,
          OP_PAUSE_POLL, OP_POLL_ONCE, OP_ADVANCE, OP_NEXTGEN, OP_GEN, OP_RESET, OP_DROPLINE, OP_DEL, OP_EXPIRE };
struct Cmd {
    std::atomic<int> state{0};   // 0 free, 1 reserved, 2 posted, 3 done
    int op = 0;
    uint64_t key = 0;
    int64_t arg = 0, res = 0;
    uint32_t ver = 0, req = 0, sf = 0;   // sf: bounded-staleness floor at the GET's start
    int kind = 0, ok = 0, okst = 1, absent = 0, okmono = 1;   // absent: the GET returned vlen 0 (a DELETE's tombstone)
};
struct TestShared {
    std::atomic<int> stop{0}, ready{0}, nmsg{0};
    CheckCounters cc;
    Stats stats[2];
    Gate gates[NGATE];
    Cmd cmd[2][NSLOT];
    char msg[20][200];
};

static Scheme g_scheme = Scheme::RACKCACHE;   // set per scenario
// true while the variant under test is Krater's committed ack (swap order, no W): read-after-write is not promised,
// so P2 is reported but not counted; P1, P3, the bytes and bounded staleness from the cut (err_stale) are
static bool g_committed = false;
static bool committed_rules(const Rules &r) { return r.put_cut_at_publish && !r.put_wait; }
// the bounded-staleness check applies to the log scheme: P (the heartbeat bound), slack for the TSC / clock rates
static bool stale_checked(const Config &c) {
    return c.rules.log && !c.rules.lease && !c.rules.revoke && c.scheme != Scheme::PRIV && c.scheme != Scheme::SHARD;
}
static std::function<void(Config &)> g_tweak;  // set per suite / scenario
static bool g_skip_writable_wait = false;     // restart scenario: start using the cache right away
static Config test_config(const Rules &r, bool emu, uint64_t slow_limit_ns) {
    Config c;
    c.scheme = g_scheme;
    c.emu_noncoherent = emu;
    c.slow_limit_ns = slow_limit_ns;
    c.hosts = 2;
    c.budget_bytes = 1 << 20;
    c.slab_bytes = 2 << 20;
    c.ai_slots = 1024;
    c.ring_lines = 1024;
    c.heat_batch = 1;
    c.rules = r;
    if (g_ro_peers) c.ro_peers = true;   // before the tweak: the ro suite's control turns it off again
    if (g_tweak) g_tweak(c);
    if (g_mono && mono_ok(c)) c.mono = true;
    if (g_xrep && c.scheme == Scheme::XTRA && c.xtra_own_freshness && c.rules.log && !c.rules.lease && !c.rules.revoke)
        c.xtra_dir_replicas = g_xrep;   // as xtra_own() (Cache::Impl::check_cfg refuses the copies elsewhere)
    if (g_force_li32) c.li16 = false;
    c.hit_clock = g_hit_clock;
    if (g_obj_read >= 0) c.obj_read_mode = g_obj_read;
    if (g_put_key_after >= 0) c.put_key_after_publish = g_put_key_after == 1;
    if (g_put_async >= 0) c.put_async = g_put_async == 1;
    c.li16_fp_bits = g_li16_fp_bits;
    c.li16_ep_bits = g_li16_ep_bits;
    return c;
}

// scenario-local config change (restored on return)
struct Tweak {
    std::function<void(Config &)> saved;
    explicit Tweak(std::function<void(Config &)> f) : saved(g_tweak) {
        auto s = saved;
        g_tweak = [s, f](Config &c) { if (s) s(c); f(c); };
    }
    ~Tweak() { g_tweak = saved; }
};

// the scenario log is appended from several threads at once (PUT / GET threads plus the main thread, e.g. M5-serial,
// A1-asyncput), so it takes a lock: a bare std::vector would race and can corrupt the heap
struct SafeLog {
    mutable std::mutex mu;
    std::vector<std::string> v;
    void push_back(std::string x) { std::lock_guard<std::mutex> g(mu); v.push_back(std::move(x)); }
    operator std::vector<std::string>() const { std::lock_guard<std::mutex> g(mu); return v; }
};
struct Env {
    SafeLog log;
    // threads mode
    std::unique_ptr<SimBackend> be;
    std::unique_ptr<Checker> ck;
    std::unique_ptr<Cache> cache;
    // procs mode
    TestShared *S = nullptr;
    size_t sim_bytes = 0;
    std::string cache_shm, sim_shm;
    std::vector<pid_t> kids;
    std::atomic<uint32_t> *ver_arr = nullptr;
    std::atomic<uint64_t> *cuts_arr = nullptr;   // procs mode: the checker's bounded-staleness ring and floors (shared)
    std::atomic<uint32_t> *floor_arr = nullptr;
    Config cfg;
    int last_kind = -1;
    uint32_t last_ver = 0;
    bool last_absent = false;   // the last get() returned vlen 0 (deleted)

    explicit Env(const Rules &r, bool emu = false, uint64_t slow_limit_ns = 60'000'000'000ULL) {
        cfg = test_config(r, emu || g_force_emu, slow_limit_ns);   // long T by default: gates hold threads for milliseconds
        if (!g_procs) {
            be.reset(new SimBackend(NKEYS, 0));
            if (g_put_delay_ns) be->set_put_delay(g_put_delay_ns);
            ck.reset(new Checker(NKEYS, be.get()));
            if (g_check_mono) ck->enable_mono(cfg.hosts);
            cfg.on_publish = [this](uint64_t k, uint32_t v) { ck->on_publish(k, v); };
            if (stale_checked(cfg)) {
                ck->set_stale_bound(cfg.log_poll_ns, cfg.log_poll_ns / 50);
                cfg.on_cut = [this](uint64_t k, uint32_t v, uint64_t c) { ck->on_cut(k, v, c); };
            }
            cache.reset(new Cache(cfg, be.get()));
            cache->start();
            if (!g_skip_writable_wait) cache->wait_writable();
            return;
        }
        static int seq = 0;
        std::string tag = "/rctest-" + std::to_string(getpid()) + "-" + std::to_string(seq++);
        cache_shm = tag + "-cache";
        sim_shm = tag + "-sim";
        cfg.shm_name = cache_shm;
        Cache::shm_create(cfg);
        const size_t off_floor = sizeof(TestShared) + 2 * (NKEYS + 1) * sizeof(std::atomic<uint32_t>);
        const size_t off_cuts = (off_floor + (NKEYS + 1) * sizeof(std::atomic<uint32_t>) + 63) / 64 * 64;
        sim_bytes = off_cuts + (NKEYS + 1) * Checker::CUT_RING * sizeof(std::atomic<uint64_t>);
        int fd = shm_open(sim_shm.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0 || ftruncate(fd, (off_t)sim_bytes)) { perror("sim shm"); exit(1); }
        char *p = (char *)mmap(nullptr, sim_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        S = new (p) TestShared;
        auto *ver = (std::atomic<uint32_t> *)(p + sizeof(TestShared));
        ver_arr = ver;
        for (uint64_t i = 0; i < 2 * (NKEYS + 1); i++) new (&ver[i]) std::atomic<uint32_t>(0);
        floor_arr = (std::atomic<uint32_t> *)(p + off_floor);
        cuts_arr = (std::atomic<uint64_t> *)(p + off_cuts);
        for (uint64_t i = 0; i <= NKEYS; i++) new (&floor_arr[i]) std::atomic<uint32_t>(0);
        for (uint64_t i = 0; i < (NKEYS + 1) * Checker::CUT_RING; i++) new (&cuts_arr[i]) std::atomic<uint64_t>(0);
        g_gates = S->gates;
        cfg.shared_stats = S->stats;
        fflush(stdout);
        for (int h = 0; h < cfg.hosts; h++) {
            pid_t pid = fork();
            if (pid == 0) child(h, ver, ver + NKEYS + 1);
            kids.push_back(pid);
        }
        wait_until([&] { return S->ready.load() == cfg.hosts; }, 10000);
    }
    ~Env() {
        if (!g_procs) { cache->stop(); return; }
        S->stop = 1;
        for (pid_t k : kids) {
            int st;
            if (!wait_until([&] { return waitpid(k, &st, WNOHANG) == k; }, 20000)) kill(k, SIGKILL);
        }
        Cache::shm_unlink(cache_shm);
        shm_unlink(sim_shm.c_str());
        munmap(S, sim_bytes);
        g_gates = g_local_gates;
    }
    [[noreturn]] void child(int h, std::atomic<uint32_t> *ver, std::atomic<uint32_t> *done) {
        SimBackend b(NKEYS, 0, ver);
        if (g_put_delay_ns) b.set_put_delay(g_put_delay_ns);
        Checker c(NKEYS, &b, done, &S->cc, cuts_arr, floor_arr);
        if (g_check_mono) c.enable_mono(cfg.hosts);
        Config cf = cfg;
        cf.local_host = h;
        cf.on_publish = [&](uint64_t k, uint32_t v) { c.on_publish(k, v); };
        if (stale_checked(cf)) {
            c.set_stale_bound(cf.log_poll_ns, cf.log_poll_ns / 50);
            cf.on_cut = [&](uint64_t k, uint32_t v, uint64_t ct) { c.on_cut(k, v, ct); };
        }
        {
            Cache ca(cf, &b);
            ca.start();
            if (!g_skip_writable_wait) ca.wait_writable();
            std::mutex msg_mu;
            size_t reported = 0;
            auto export_errors = [&] {   // copy new checker messages of this child to the shared log
                std::lock_guard<std::mutex> g(msg_mu);
                auto m = c.first_errors();
                for (; reported < m.size(); reported++) {
                    int i = S->nmsg++;
                    if (i < 20) snprintf(S->msg[i], sizeof S->msg[i], "%s", m[reported].c_str());
                }
            };
            std::vector<std::thread> srv;
            for (int sl = 0; sl < NSLOT; sl++)
                srv.emplace_back([&, sl] {
                    Cmd &x = S->cmd[h][sl];
                    while (!S->stop.load()) {
                        if (x.state.load() != 2) { std::this_thread::sleep_for(std::chrono::microseconds(50)); continue; }
                        switch (x.op) {
                        case OP_GET: {
                            Client cl(&ca, h);
                            char buf[MAX_OUT];
                            Kind kind;
                            x.req = c.begin(x.key);
                            x.sf = c.stale_begin(x.key);
                            const uint32_t mf = c.mono_begin(h, x.key);
                            uint32_t vlen = cl.get(x.key, buf, &kind);
                            x.ok = c.verify(x.key, x.req, buf, vlen, h, kind);
                            x.absent = vlen == 0;
                            memcpy(&x.ver, buf + 8, 4);
                            x.okmono = c.verify_mono(h, x.key, mf, x.ver, kind);
                            x.okst = c.verify_stale(x.key, x.sf, x.ver, h, kind);
                            x.kind = kind;
                            break;
                        }
                        case OP_PUT: {
                            Client cl(&ca, h);
                            x.ver = cl.put(x.key);
                            if (x.ver) c.put_done(x.key, x.ver);   // 0 = failed PUT
                            break;
                        }
                        case OP_DEL: {   // a DELETE's version (the tombstone) completes like a PUT's
                            Client cl(&ca, h);
                            x.ver = cl.del(x.key);
                            if (x.ver) c.put_done(x.key, x.ver);
                            break;
                        }
                        case OP_EXPIRE: { Client cl(&ca, h); x.ver = cl.expire(x.key); break; }   // no version
                        case OP_EVICT: x.res = ca.owner_evict(x.key); break;
                        case OP_CACHED: x.res = ca.owner_cached(x.key); break;
                        case OP_OFFSET: x.res = ca.owner_offset(x.key); break;
                        case OP_PREFETCH: ca.emu_prefetch(h, (int)(x.arg >> 48), (uint64_t)x.arg & ((1ULL << 48) - 1)); break;
                        case OP_DROP: ca.emu_drop_all(h); break;
                        case OP_PAUSE_EPOCH: ca.pause_epoch(h, x.arg != 0); break;
                        case OP_UNREACH: ca.set_unreachable(h, x.arg != 0); break;
                        case OP_PAUSE_POLL: ca.pause_poller(h, x.arg != 0); break;
                        case OP_POLL_ONCE: ca.poll_once(h); break;
                        case OP_ADVANCE: x.res = ca.owner_advance(h, (uint64_t)x.arg); break;
                        case OP_NEXTGEN: x.res = ca.owner_next_gen(h); break;
                        case OP_GEN: x.res = ca.owner_gen(x.key); break;
                        case OP_RESET: ca.owner_reset(h); break;
                        case OP_DROPLINE: ca.emu_drop_line(h, (int)(x.arg >> 48), (uint64_t)x.arg & ((1ULL << 48) - 1)); break;
                        }
                        export_errors();
                        x.state = 3;
                    }
                });
            S->ready++;
            for (auto &t : srv) t.join();
            ca.stop();
            export_errors();
        }
        fflush(stdout);
        _exit(0);
    }
    Cmd &run(int h, int op, uint64_t key, int64_t arg = 0) {
        for (;;)
            for (int sl = 0; sl < NSLOT; sl++) {
                Cmd &x = S->cmd[h][sl];
                int zero = 0;
                if (!x.state.compare_exchange_strong(zero, 1)) continue;
                x.op = op; x.key = key; x.arg = arg;
                x.state = 2;
                wait_until([&] { return x.state.load() == 3; }, 60000);
                return x;   // caller copies what it needs, then free()
            }
    }
    static void release(Cmd &x) { x.state = 0; }

    int owner_of(uint64_t k) { return Cache::owner_of_static(k, cfg.hosts); }
    static const char *kn(int k) { static const char *n[] = {"FAST", "SLOW", "MISS"}; return n[k]; }
    // one GET by a fresh client of host h, checked; returns the version observed
    uint32_t get(int h, uint64_t k, const char *tag) {
        uint32_t v, req, sf;
        int kind;
        bool ok, okst, absent, okmono = true;
        if (!g_procs) {
            Client c(cache.get(), h);
            char buf[MAX_OUT];
            Kind kd;
            req = ck->begin(k);
            sf = ck->stale_begin(k);
            const uint32_t mf = ck->mono_begin(h, k);
            uint32_t vlen = c.get(k, buf, &kd);
            ok = ck->verify(k, req, buf, vlen, h, kd);
            absent = vlen == 0;
            memcpy(&v, buf + 8, 4);
            okst = ck->verify_stale(k, sf, v, h, kd);
            okmono = ck->verify_mono(h, k, mf, v, kd);
            kind = kd;
        } else {
            Cmd &x = run(h, OP_GET, k);
            v = x.ver; req = x.req; sf = x.sf; kind = x.kind; ok = x.ok; okst = x.okst; absent = x.absent; okmono = x.okmono;
            release(x);
        }
        last_kind = kind;
        last_ver = v;
        last_absent = absent;
        const char *note = "";
        if (!ok && g_committed && v < req) note = "  (older than the last ack: no RAW for committed, not counted)";
        else if (!ok) note = "  <-- CHECKER ERROR";
        log.push_back(std::string(tag) + ": host" + std::to_string(h) + " GET -> v" + std::to_string(v) + (absent ? " absent" : "") +
                      " via " + kn(kind) + " (req v" + std::to_string(req) + (sf ? ", stale floor v" + std::to_string(sf) : "") + ")" +
                      note + (okst ? "" : "  <-- STALE (cut more than P before the GET)") +
                      (okmono ? "" : "  <-- P6 (older than a version this host already returned)"));
        return v;
    }
    uint32_t put(int h, uint64_t k, const char *tag) {
        uint32_t v;
        if (!g_procs) {
            Client c(cache.get(), h);
            v = c.put(k);
            if (v) ck->put_done(k, v);   // 0 = failed PUT
        } else {
            Cmd &x = run(h, OP_PUT, k);
            v = x.ver;
            release(x);
        }
        log.push_back(std::string(tag) + ": host" + std::to_string(h) + (v ? " PUT -> v" + std::to_string(v) : " PUT FAILED"));
        return v;
    }
    // DELETE (its version completes like a PUT's) and EXPIRE (TTL: no version; returns 1, 0 = failed)
    uint32_t del(int h, uint64_t k, const char *tag) {
        uint32_t v;
        if (!g_procs) {
            Client c(cache.get(), h);
            v = c.del(k);
            if (v) ck->put_done(k, v);
        } else {
            Cmd &x = run(h, OP_DEL, k);
            v = x.ver;
            release(x);
        }
        log.push_back(std::string(tag) + ": host" + std::to_string(h) + (v ? " DELETE -> v" + std::to_string(v) + " (tombstone)" : " DELETE FAILED"));
        return v;
    }
    uint32_t expire(int h, uint64_t k, const char *tag) {
        uint32_t v;
        if (!g_procs) {
            Client c(cache.get(), h);
            v = c.expire(k);
        } else {
            Cmd &x = run(h, OP_EXPIRE, k);
            v = x.ver;
            release(x);
        }
        log.push_back(std::string(tag) + ": host" + std::to_string(h) + (v ? " EXPIRE done" : " EXPIRE FAILED"));
        return v;
    }
    uint32_t put_quiet(int h, uint64_t k) {   // a PUT without a log line (safe from a second thread)
        uint32_t v;
        if (!g_procs) {
            Client c(cache.get(), h);
            v = c.put(k);
            if (v) ck->put_done(k, v);
        } else {
            Cmd &x = run(h, OP_PUT, k);
            v = x.ver;
            release(x);
        }
        return v;
    }
    int64_t owner_op(int op, uint64_t k) {
        Cmd &x = run(owner_of(k), op, k);
        int64_t r = x.res;
        release(x);
        return r;
    }
    bool owner_evict(uint64_t k) { return g_procs ? owner_op(OP_EVICT, k) != 0 : cache->owner_evict(k); }
    bool owner_cached(uint64_t k) { return g_procs ? owner_op(OP_CACHED, k) != 0 : cache->owner_cached(k); }
    int64_t owner_offset(uint64_t k) { return g_procs ? owner_op(OP_OFFSET, k) : cache->owner_offset(k); }
    void emu_drop_all(int h) {
        if (!g_procs) return cache->emu_drop_all(h);
        release(run(h, OP_DROP, 0));
    }
    void emu_prefetch(int h, int o, uint64_t off) {
        if (!g_procs) return cache->emu_prefetch(h, o, off);
        release(run(h, OP_PREFETCH, 0, (int64_t)((uint64_t)o << 48 | off)));
    }
    Stats &stats(int h) { return g_procs ? S->stats[h] : cache->stats(h); }
    void pause_epoch(int h, bool p) {
        if (!g_procs) return cache->pause_epoch(h, p);
        release(run(h, OP_PAUSE_EPOCH, 0, p));
    }
    void set_unreachable(int h, bool u) {
        if (!g_procs) return cache->set_unreachable(h, u);
        release(run(h, OP_UNREACH, 0, u));
    }
    void pause_poller(int h, bool p) {
        if (!g_procs) return cache->pause_poller(h, p);
        release(run(h, OP_PAUSE_POLL, 0, p));
    }
    void poll_once(int h) {
        if (!g_procs) return cache->poll_once(h);
        release(run(h, OP_POLL_ONCE, 0));
    }
    int64_t host_op(int h, int op, int64_t arg = 0) {
        Cmd &x = run(h, op, 0, arg);
        int64_t r = x.res;
        release(x);
        return r;
    }
    uint32_t owner_advance(int o, uint64_t n) { return g_procs ? (uint32_t)host_op(o, OP_ADVANCE, (int64_t)n) : cache->owner_advance(o, n); }
    uint32_t owner_next_gen(int o) { return g_procs ? (uint32_t)host_op(o, OP_NEXTGEN) : cache->owner_next_gen(o); }
    uint32_t owner_gen(uint64_t k) { return g_procs ? (uint32_t)owner_op(OP_GEN, k) : cache->owner_gen(k); }
    void owner_reset(int o) {
        if (!g_procs) return cache->owner_reset(o);
        host_op(o, OP_RESET);
    }
    void emu_drop_line(int h, int o, uint64_t off) {
        if (!g_procs) return cache->emu_drop_line(h, o, off);
        host_op(h, OP_DROPLINE, (int64_t)((uint64_t)o << 48 | off));
    }
    uint32_t backend_version(uint64_t k) { return g_procs ? ver_arr[k].load() : be->current(k); }
    // committed (g_committed): P2 is not counted (no read-after-write promised); bounded staleness is
    uint64_t errors() {
        const uint64_t m = g_check_mono ? (g_procs ? S->cc.err_mono.load() : ck->mono_errors()) : 0;   // P6 (mono suite)
        if (g_procs)
            return S->cc.err_p1 + (g_committed ? 0 : S->cc.err_p2.load()) + S->cc.err_p3 + S->cc.err_bytes + S->cc.err_stale + m;
        return ck->errors() - (g_committed ? ck->p2_errors() : 0) + m;
    }
    std::vector<std::string> first_errors() {
        std::vector<std::string> v;
        if (!g_procs) v = ck->first_errors();
        else
            for (int i = 0; i < std::min(20, S->nmsg.load()); i++) v.push_back(S->msg[i]);
        if (g_committed) v.erase(std::remove_if(v.begin(), v.end(), [](const std::string &m) { return m.find(": P2:") != std::string::npos; }), v.end());
        return v;
    }
    void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }   // let handlers drain
};

static uint64_t key_owned_by(Env &c, int o, uint64_t after = 0, bool one_line = false) {
    for (uint64_t k = after + 1; k < 1024; k++)
        if (c.owner_of(k) == o && (!one_line || (obj_lines(value_len(k, 0)) == 1 && obj_lines(value_len(k, 1)) == 1)))
            return k;
    abort();
}
// a key of owner o whose first version is a two-line object (the generation-wrap scenarios tear line 0 from line 1)
static uint64_t key_two_lines(Env &c, int o, uint64_t after = 0, bool v1_too = false) {
    for (uint64_t k = after + 1; k < 1024; k++)
        if (c.owner_of(k) == o && obj_lines(value_len(k, 0)) == 2 && (!v1_too || obj_lines(value_len(k, 1)) == 2)) return k;
    abort();
}

struct Result { uint64_t errors; bool reproduced; std::vector<std::string> log, errs; };

// Scenario 1 (tombstone; fails in fixed-notomb): the reader's slow path reads the key's AI entry, then a whole PUT runs
// (its revoke reaches the reader host while the slow path is in flight), then the reader adopts the old entry;
// a GET that starts after the PUT hits the old value on the fast path. The entry is read before the PUT starts, so
// the tombstone is needed with or without unpublish_on_put.
static Result scenario_tomb(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    wait_until([&] { return e.owner_cached(K); });
    Gate &gR = new_gate(Hook::SlowAfterAiRead, 1, K);
    arm(gR);
    std::thread t2([&] { e.get(1, K, "T2 (overlaps PUT)"); });
    bool ok = wait_until([&] { return gR.reached.load(); });
    e.put(0, K, "T1");
    gR.released = true;
    t2.join();
    e.settle();
    e.get(1, K, "T3 (after PUT)");
    disarm_all();
    ok &= !gR.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// Scenario 2 (bump2; fails in fixed-nobump2): a fill token taken between the PUT's acks and its backend write carries
// the old value; after the PUT returns and its object is evicted, the stale fill is admitted.
static Result scenario_bump2(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "preload");
    Gate &gP = new_gate(Hook::PutAfterAcks, 0, K), &gF = new_gate(Hook::MissBeforeSendFill, 1, K);
    arm(gP);
    std::thread t1([&] { e.put(0, K, "T1"); });
    bool ok = wait_until([&] { return gP.reached.load(); });
    arm(gF);
    std::thread t2([&] { e.get(1, K, "T2 (overlaps PUT)"); });
    ok &= wait_until([&] { return gF.reached.load(); });
    gP.released = true;
    t1.join();
    ok &= e.owner_evict(K);
    e.log.push_back("owner evicts K");
    gF.released = true;
    t2.join();
    e.settle();
    e.get(1, K, "T3 (after PUT)");
    disarm_all();
    ok &= !gP.timed_out && !gF.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// Scenario 3 (pending; fails in fixed-nopending): K is not cached; a miss fill during the PUT is admitted with the bumped
// kg; a later slow path reads that entry, the PUT completes, the reader adopts it; the next GET hits the old value.
static Result scenario_pending(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    wait_until([&] { return e.owner_cached(K); });
    bool ok = e.owner_evict(K);
    e.log.push_back("owner evicts K");
    Gate &gP = new_gate(Hook::PutAfterAcks, 0, K), &gR = new_gate(Hook::SlowAfterAiRead, 1, K);
    arm(gP);
    std::thread t1([&] { e.put(0, K, "T1"); });
    ok &= wait_until([&] { return gP.reached.load(); });
    e.get(1, K, "T2 (overlaps PUT)");
    e.settle();
    e.log.push_back(std::string("owner cached K during PUT: ") + (e.owner_cached(K) ? "yes" : "no"));
    arm(gR);
    std::thread t3([&] { e.get(1, K, "T3 (overlaps PUT)"); });
    ok &= wait_until([&] { return gR.reached.load(); });
    gP.released = true;
    t1.join();
    gR.released = true;
    t3.join();
    e.settle();
    e.get(1, K, "T4 (after PUT)");
    disarm_all();
    ok &= !gP.timed_out && !gR.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// Scenario 4 (install; fails in the *-installfirst variants: the local-index entry installed before the flush):
// the owner rewrites a slot; while the line holds the new gen but the old payload, the reader host's cache
// picks it up (prefetch). Thread T0's slow path reads the new index entry and installs it in the local index;
// if it installs BEFORE flushing, thread T1 of the same host uses the new gen on the stale cached line and
// returns the old payload. Needs the software non-coherent cache (coherent DRAM never serves a stale line).
static Result scenario_install(const Rules &r) {
    Env e(r, true);
    uint64_t K = key_owned_by(e, 0, 0, true), K2 = key_owned_by(e, 0, K, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    int64_t X = e.owner_offset(K);
    e.get(1, K, "preload");
    e.put(0, K, "PUT");                       // v1 goes to a new slot; slot X is freed
    ok &= e.owner_evict(K);
    e.get(0, K2, "filler");                   // takes the slot the PUT used, so K comes back into X
    ok &= wait_until([&] { return e.owner_cached(K2); }) && e.owner_offset(K2) != X;
    e.log.push_back("owner evicts K; another key takes the PUT's slot; slot X=" + std::to_string(X) + " is free");
    Gate &gW = new_gate(Hook::OwnerAfterGenStore, 0, K), &gI = new_gate(Hook::SlowAfterInstall, 1, K);
    arm(gW);
    std::thread tw([&] { e.get(0, K, "owner host GET (re-admits K)"); });
    ok &= wait_until([&] { return gW.reached.load(); });
    e.emu_drop_all(1);
    e.emu_prefetch(1, 0, (uint64_t)X);
    e.log.push_back("owner stored the new gen of line 0 only; host1 prefetches that line (new gen, old payload)");
    gW.released = true;
    tw.join();
    ok &= wait_until([&] { return e.owner_offset(K) == X; });
    arm(gI);
    std::thread t0([&] { e.get(1, K, "T0 (slow path)"); });
    ok &= wait_until([&] { return gI.reached.load(); });
    e.get(1, K, "T1 (same host, after PUT)");
    gI.released = true;
    t0.join();
    disarm_all();
    ok &= !gW.timed_out && !gI.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// Scenario 5 (deadline: the slow-path time limit T): like scenario 1, but the reader's slow path stalls
// 3T between reading the AI entry and installing it, so its tombstone has expired. Without the deadline check
// the stale entry is adopted; with it the slow path restarts and reads the new entry.
static Result scenario_deadline(const Rules &r) {
    const uint64_t T = 20'000'000;
    Env e(r, false, T);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    wait_until([&] { return e.owner_cached(K); });
    Gate &gR = new_gate(Hook::SlowAfterAiRead, 1, K);
    arm(gR);
    std::thread t2([&] { e.get(1, K, "T2 (overlaps PUT, stalls 3T after reading the AI entry)"); });
    bool ok = wait_until([&] { return gR.reached.load(); });
    e.put(0, K, "T1");
    std::this_thread::sleep_for(std::chrono::nanoseconds(3 * T));
    gR.released = true;
    t2.join();
    e.settle();
    e.get(1, K, "T3 (after PUT)");
    disarm_all();
    ok &= !gR.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// Scenario 6 (slowput; motivates the rule unpublish_on_put): the PUT stalls
// longer than T between its revokes and publishing the new value. Tombstones age from the revoke, so a slow
// path that starts after they expired still finds the OLD AI entry, passes the (now absent) tombstone check
// and adopts it; after the PUT returns, the fast path serves the old value. Removing the AI entry when the
// PUT starts leaves nothing stale to adopt.
static Result scenario_slowput(const Rules &r) {
    const uint64_t T = 20'000'000;
    Env e(r, false, T);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "preload");
    Gate &gP = new_gate(Hook::PutAfterAcks, 0, K);
    arm(gP);
    std::thread t1([&] { e.put(0, K, "T1 (stalls 3T after its acks)"); });
    bool ok = wait_until([&] { return gP.reached.load(); });
    std::this_thread::sleep_for(std::chrono::nanoseconds(3 * T));   // host1's tombstone for K expires
    e.get(1, K, "T2 (overlaps PUT)");
    gP.released = true;
    t1.join();
    e.settle();
    e.get(1, K, "T3 (after PUT)");
    disarm_all();
    ok &= !gP.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// Scenario 7 (privwriter, PRIV baseline): host 1 PUTs K; before it installs the value it wrote into its private cache,
// host 0 PUTs K again (to completion). The writer must install with the token captured when its own PUT
// completed; reading the token at install time admits the older value and the next GET on host 1 returns it.
static Result scenario_privwriter(const Rules &r) {
    g_scheme = Scheme::PRIV;
    Result res;
    {
        Env e(r);
        uint64_t K = key_owned_by(e, 0);
        e.get(1, K, "preload");
        Gate &gW = new_gate(Hook::PrivWriterBeforeInstall, 1, K);
        arm(gW);
        std::thread t1([&] { e.put(1, K, "T1 (host1 writes, stalls before installing its own value)"); });
        bool ok = wait_until([&] { return gW.reached.load() != 0; });
        e.put(0, K, "T2 (host0 writes again, completes)");
        gW.released = 1;
        t1.join();
        e.settle();
        e.get(1, K, "T3 (after both PUTs)");
        disarm_all();
        ok &= !gW.timed_out;
        res = {e.errors(), ok, e.log, e.first_errors()};
    }
    g_scheme = Scheme::RACKCACHE;
    return res;
}

static std::string drops(Env &e, int h);
// ================================================================ lease scenarios (--update lease)
// The lease suite runs with T = TL (long, so that the gates' milliseconds do not expire leases by accident).
constexpr uint64_t TL = 400'000'000;
static void nap(uint64_t ns) { std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); }
static void short_settle() { nap(20'000'000); }   // let handlers drain without getting near T

// L1 (fails in lease-at-install): the reader's slow path reads the OLD AI entry, then the whole PUT runs and returns
// (T after its cut), then the reader installs the entry. If its lease started at install time the entry is
// fresh and the next GET, which starts after the PUT returned, hits the old value. Started at the AI read
// (before the cut), the lease has already expired at install.
static Result lease_atinstall(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    Gate &gR = new_gate(Hook::SlowAfterAiRead, 1, K);
    arm(gR);
    std::thread t2([&] { e.get(1, K, "T2 (slow path reads the old AI entry, stalls across the PUT)"); });
    ok &= wait_until([&] { return gR.reached.load(); });
    e.put(0, K, "T1 (returns T after its cut)");
    gR.released = true;
    t2.join();
    e.get(1, K, "T3 (after PUT)");
    disarm_all();
    ok &= !gR.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// L2 (fails in lease-nowait; requirement "a GET that starts after the PUT returned reads the new value"): host 1 holds
// a fresh entry; the PUT returns; a GET right after must not hit the old object. Only the T wait guarantees it.
static Result lease_putreturn(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs a fresh entry)");
    e.put(0, K, "T1");
    e.get(1, K, "T2 (right after the PUT returned)");
    return {e.errors(), ok, e.log, e.first_errors()};
}

// L3 (fails in lease-nobump2 and lease-nofilltok): a miss takes its fill token after the PUT's first kg bump and
// reads the backend before the PUT writes it; the fill arrives after the PUT returned (pending is over) and the
// PUT's object was evicted. Only the second bump (token != kg) or the token check itself rejects it.
static Result swap_stalefill(const Rules &r);
static Result lease_stalefill(const Rules &r) {
    if (r.put_cut_at_publish) return swap_stalefill(r);   // swap order: no bump at the start (see below)
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    Gate &gP = new_gate(Hook::PutAfterAcks, 0, K), &gF = new_gate(Hook::MissBeforeSendFill, 1, K);
    arm(gP);
    std::thread t1([&] { e.put(0, K, "T1"); });
    ok &= wait_until([&] { return gP.reached.load(); });
    arm(gF);
    std::thread t2([&] { e.get(1, K, "T2 (miss during the PUT: token, then the old backend value)"); });
    ok &= wait_until([&] { return gF.reached.load(); });
    gP.released = true;
    t1.join();
    ok &= e.owner_evict(K);
    e.log.push_back("owner evicts K");
    gF.released = true;
    t2.join();
    short_settle();
    e.log.push_back(drops(e, 0));
    e.get(1, K, "T3 (after PUT)");
    disarm_all();
    ok &= !gP.timed_out && !gF.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// L4 (fails in lease-nopending; requirement "a fill during a PUT is rejected"): K is not cached; after the PUT's cut a
// miss fill is admitted (it carries the bumped kg); more than T later a slow path reads that entry, the PUT completes and
// returns at once (T after its cut has passed), the reader installs the entry with a fresh lease and the next
// GET hits the old value. Only "no fills while a PUT is in progress" prevents it.
static Result lease_fillduringput(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    // not cached when the PUT starts: an object the PUT unpublished stays in the owner's table until the PUT ends and
    // would also block fills ("already cached")
    ok &= e.owner_evict(K);
    e.log.push_back("owner evicts K");
    uint64_t drop0 = e.stats(0).fill_drop_pending;
    Gate &gP = new_gate(Hook::PutAfterAcks, 0, K), &gR = new_gate(Hook::SlowAfterAiRead, 1, K);
    arm(gP);
    std::thread t1([&] { e.put(0, K, "T1"); });
    ok &= wait_until([&] { return gP.reached.load(); });
    e.get(1, K, "T2 (miss during the PUT, sends a fill)");
    short_settle();
    e.log.push_back(std::string("owner cached K during PUT: ") + (e.owner_cached(K) ? "yes" : "no") +
                    ", fills dropped as pending: " + std::to_string(e.stats(0).fill_drop_pending - drop0));
    nap(TL + 50'000'000);   // T after the cut has passed
    arm(gR);
    std::thread t3([&] { e.get(1, K, "T3 (slow path reads the AI entry during the PUT)"); });
    ok &= wait_until([&] { return gR.reached.load(); });
    gP.released = true;
    t1.join();
    gR.released = true;
    t3.join();
    e.get(1, K, "T4 (after PUT)");
    disarm_all();
    ok &= !gP.timed_out && !gR.timed_out;
    if (r.pending) ok &= e.stats(0).fill_drop_pending > drop0;   // the rejection happened
    return {e.errors(), ok, e.log, e.first_errors()};
}

// L5 (requirement "a hot key rewritten in place is read new after T, through one slow path"; the
// "stale forever" problem): host 1 keeps hitting K; the owner rewrites K; afterwards host 1 must read the new
// value, through a slow path. Without leases (and without revocation) it keeps hitting the old object forever.
static Result lease_hotrewrite(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0");
    e.get(1, K, "T1 (hot: fast path)");
    e.put(0, K, "T2 (owner rewrites K)");
    e.get(1, K, "T3 (after the PUT)");
    if (r.lease) ok &= e.last_kind == SLOW;
    nap(2 * TL);
    e.get(1, K, "T4 (2T later)");
    return {e.errors(), ok, e.log, e.first_errors()};
}

// L6 (requirement "the slow path notices a stopped epoch thread"): host 1's epoch thread stops; host 1 holds an
// entry for K; the PUT on K returns (T after its cut, but host 1's epoch did not move). A slow path on host 1
// (another key) sees with the TSC that the epoch is more than T behind and advances it by 2, so K's entry
// expires and the next GET of K reads the new value.
static Result lease_epochstall(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0), K2 = key_owned_by(e, 0, K);
    e.get(0, K, "preload");
    e.get(0, K2, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K) && e.owner_cached(K2); });
    e.get(1, K, "T0 (host1 installs K)");
    e.pause_epoch(1, true);
    e.log.push_back("host1's epoch thread stops");
    e.put(0, K, "T1");
    nap(TL / 4);
    uint64_t st0 = e.stats(1).epoch_stall;
    e.get(1, K2, "T2 (host1 slow path on another key)");
    e.get(1, K, "T3 (after PUT)");
    if (r.lease_tsc) ok &= e.stats(1).epoch_stall > st0;
    e.pause_epoch(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// L7 (a known limitation of leases): as L6 but host 1 runs no slow path, only hits: nothing checks its epoch, so it
// keeps reading the old value. "Stale at most T" assumes the epoch thread keeps ticking.
static Result lease_stall_limit(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.pause_epoch(1, true);
    e.log.push_back("host1's epoch thread stops");
    e.put(0, K, "T1");
    e.get(1, K, "T2 (after PUT, hit only)");
    e.pause_epoch(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// L8 (design: PUTs go through the owner; owner unreachable -> the PUT fails): host 0 (owner) stops reading its
// rings; host 1's PUT times out and returns 0; the backend is not written; reads stay consistent.
static Result lease_unreachable(const Rules &r) {
    auto saved = g_tweak;
    g_tweak = [saved](Config &c) { if (saved) saved(c); c.put_timeout_ns = 300'000'000; };
    Result res;
    {
        Env e(r);
        uint64_t K = key_owned_by(e, 0);
        e.get(1, K, "preload");
        bool ok = wait_until([&] { return e.owner_cached(K); });
        uint32_t b0 = e.backend_version(K);
        e.set_unreachable(0, true);
        e.log.push_back("owner host0 unreachable");
        uint32_t v = e.put(1, K, "T1 (host1 PUT)");
        ok &= v == 0 && e.backend_version(K) == b0 && e.stats(1).put_failed >= 1;
        e.log.push_back("backend version after the failed PUT: v" + std::to_string(e.backend_version(K)));
        e.get(1, K, "T2");
        e.set_unreachable(0, false);
        e.log.push_back("owner reachable again (the queued PUT request may still execute)");
        short_settle();
        e.get(1, K, "T3");
        res = {e.errors(), ok, e.log, e.first_errors()};
    }
    g_tweak = saved;
    return res;
}

// U1 (an unreachable owner's keys stay readable): host 0 stops reading its rings; host 1's GET of a key of host 0
// that host 1 has no entry for misses, gets no fill token within tok_timeout_ns, reads the backend and does not fill;
// the next miss on host 0 skips the token request (backoff). Without the token timeout such a GET would hang forever.
static Result unreachable_get(const Rules &r) {
    Tweak tw([](Config &c) { c.tok_timeout_ns = 50'000'000; c.tok_backoff_ns = 2'000'000'000; });
    Env e(r);
    uint64_t K = key_owned_by(e, 0), K2 = key_owned_by(e, 0, K);
    uint64_t to0 = e.stats(1).tok_timeout;
    e.set_unreachable(0, true);
    e.log.push_back("owner host0 unreachable");
    auto t0 = std::chrono::steady_clock::now();
    e.get(1, K, "T1 (miss: no token in time, backend)");
    e.get(1, K2, "T2 (miss: owner skipped, backend)");
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    e.log.push_back("two GETs took " + std::to_string((int)ms) + " ms; token timeouts on host1: " +
                    std::to_string(e.stats(1).tok_timeout - to0));
    // with kg in CXL (kg_cxl) a miss never waits for the owner: no token timeouts at all
    bool ok = e.last_kind == MISS && ms < 1000 && !e.owner_cached(K) &&
              e.stats(1).tok_timeout - to0 == (r.kg_cxl ? 0u : 2u);
    e.set_unreachable(0, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// L9 (design: an owner (re)starting waits T before its first write): fills within T of the start are dropped.
static Result lease_restart(const Rules &r) {
    g_skip_writable_wait = true;
    Result res;
    {
        Env e(r);
        uint64_t K = key_owned_by(e, 0);
        e.get(0, K, "T0 (right after the owner started)");
        short_settle();
        bool cached_early = e.owner_cached(K);
        e.log.push_back(std::string("cached within T of the start: ") + (cached_early ? "yes" : "no") +
                        ", fills dropped: " + std::to_string(e.stats(0).fill_drop_restart));
        nap(TL + 50'000'000);
        e.get(0, K, "T1 (after T)");
        bool ok = wait_until([&] { return e.owner_cached(K); });
        if (r.restart_wait) ok &= !cached_early && e.stats(0).fill_drop_restart >= 1;
        res = {e.errors(), ok, e.log, e.first_errors()};
    }
    g_skip_writable_wait = false;
    return res;
}

// L10 (design: a GET paused longer than T is discarded and retried): host 1's fast path stalls after reading
// its entry, across a whole PUT; when it resumes, its GET spanned >= 2 epochs, so the hit is retried and reads
// the new value. (Returning the old value would be legal here, the GET started before the PUT; the bracket is
// what bounds a paused reader in general, e.g. against a generation reused after a counter round.)
static Result lease_bracket(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    uint64_t rt0 = e.stats(1).lease_retry;
    Gate &gF = new_gate(Hook::FastAfterLiRead, 1, K);
    arm(gF);
    std::thread t2([&] { e.get(1, K, "T2 (fast path stalls after reading its entry, across the PUT)"); });
    ok &= wait_until([&] { return gF.reached.load(); });
    e.put(0, K, "T1");
    gF.released = true;
    t2.join();
    disarm_all();
    if (r.lease_bracket) ok &= e.stats(1).lease_retry > rt0;
    e.log.push_back("lease retries on host1: " + std::to_string(e.stats(1).lease_retry - rt0));
    ok &= !gF.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// ================================================================ log scenarios (the owner log, polled every P)
// The log suite runs with P = W = TP (long, so that the gates' milliseconds do not age heartbeats by accident). Host 1's
// polling thread is paused in every scenario below and its polls are forced with poll_once(), so that each interleaving
// is reproduced step by step. A poll_once() right before a GET makes host 1's heartbeat fresh.
constexpr uint64_t TP = 200'000'000;

// G1 (fails in log-nowait): host 1 holds a fresh entry and a fresh heartbeat; the PUT returns; a GET right after must not
// hit the old object. Only the W wait (the heartbeat is older than P when the PUT returns) guarantees it.
static Result log_putreturn(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    e.log.push_back("host1 polls (fresh heartbeat); its poller stays paused");
    e.put(0, K, "T1");
    e.get(1, K, "T2 (right after the PUT returned)");
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// M4-tsbefore (heartbeat timing with several owners): host 1's poll round has read the tail(s) -- no record
// for K yet -- and stalls; meanwhile a PUT of owner 0's K completes its cut; then the round finishes. With the timestamp
// taken before the first tail, the heartbeat predates the cut, so when the PUT returns (W after its cut) it is older than
// P and the hit goes to the slow path. Taken after the tail reads, the heartbeat postdates the cut although the round did
// not see K's record: the hit returns the old value after the PUT returned.
static Result log_tsbefore(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    Gate &gP = new_gate(Hook::PollAfterTails, 1, 0);
    arm(gP);
    std::thread tp([&] { e.poll_once(1); });
    ok &= wait_until([&] { return gP.reached.load(); });
    e.log.push_back("host1's poll round has read the tails (no record for K) and stalls");
    Gate &gC = new_gate(Hook::PutAfterBump1, 0, K);
    arm(gC);
    std::thread t1([&] { e.put(0, K, "T1 (its cut falls inside host1's stalled round)"); });
    ok &= wait_until([&] { return gC.reached.load(); });
    gC.released = true;
    // a margin after the cut, so that the late timestamp is clearly younger than P when the PUT returns even with the
    // procs mode's command round trips
    std::this_thread::sleep_for(std::chrono::nanoseconds(TP / 4));
    gP.released = true;
    tp.join();
    e.log.push_back("PUT's cut done; 50 ms later host1's round finishes and publishes its heartbeat");
    t1.join();
    e.get(1, K, "T2 (right after the PUT returned)");
    disarm_all();
    ok &= !gP.timed_out && !gC.timed_out;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// M4-waitfromcut: the PUT unpublishes and stalls longer than W before its append; host 1 polls during the
// stall (no record yet: fresh heartbeat). W counted from the cut (record and tail written back) makes the PUT return W
// after the append, when that heartbeat is older than P; counted from the PUT's start, it returns right after the append
// and host 1's next hit returns the old value.
static Result log_waitfromcut(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    Gate &gS = new_gate(Hook::PutAfterStep1, 0, K);
    arm(gS);
    std::thread t1([&] { e.put(0, K, "T1 (unpublishes, stalls longer than W before its append)"); });
    ok &= wait_until([&] { return gS.reached.load(); });
    std::this_thread::sleep_for(std::chrono::nanoseconds(TP + TP / 4));
    e.poll_once(1);
    e.log.push_back("host1 polls during the stall (no record yet; fresh heartbeat)");
    gS.released = true;
    t1.join();
    e.get(1, K, "T2 (right after the PUT returned)");
    disarm_all();
    ok &= !gS.timed_out;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// M5-serial: two PUTs of K at owner 0. PUT#0 cuts and stalls; host 1 polls
// (applies PUT#0's record: fresh heartbeat); PUT#1 cuts, writes the backend first (v1) and publishes; host 1's slow path
// reads v1's AI entry and stalls; PUT#0 writes the backend (the later version), publishes and returns W after its cut;
// host 1 installs v1 (no record applied since its s0) and its next hit returns v1 although PUT#0 returned. With the
// owner running its PUTs one at a time, PUT#1 cannot start before PUT#0 is done.
static Result put_serial(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    uint32_t g0 = e.owner_gen(K);
    e.poll_once(1);
    Gate &gA = new_gate(Hook::PutAfterBump1, 0, K);   // the first PUT to get there (PUT#0) stalls after its cut
    arm(gA);
    std::thread t0([&] { e.put(0, K, "PUT#0 (stalls after its cut; returns W after it)"); });
    ok &= wait_until([&] { return gA.reached.load(); });
    auto tA = std::chrono::steady_clock::now();
    auto ms = [&] { return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tA).count()) + " ms: "; };
    std::this_thread::sleep_for(std::chrono::nanoseconds(TP / 4));   // host 1's heartbeat clearly after PUT#0's cut
    e.poll_once(1);
    e.log.push_back(ms() + "host1 polls: applies PUT#0's record, fresh heartbeat");
    std::thread t1([&] { e.put(0, K, "PUT#1"); });
    bool p1 = wait_until([&] { return e.owner_cached(K) && e.owner_gen(K) != g0; }, 500);
    e.log.push_back(ms() + (p1 ? "PUT#1 cut, wrote the backend and published while PUT#0 stalls"
                       : "PUT#1 waits for PUT#0 (owner runs one PUT at a time)"));
    Gate &gR = new_gate(Hook::SlowAfterAiRead, 1, K);
    arm(gR);
    std::thread t2([&] { e.get(1, K, "T2 (host1's slow path reads the AI entry, stalls)"); });
    ok &= wait_until([&] { return gR.reached.load(); });
    e.log.push_back(ms() + "T2 stalled; releasing PUT#0");
    gA.released = true;
    t0.join();
    e.log.push_back(ms() + "PUT#0 returned");
    gR.released = true;
    t2.join();
    e.log.push_back(ms() + "T2 done");
    Stats &s1 = e.stats(1);
    uint64_t r0 = s1.log_refused, n0 = s1.slow_noentry, d0 = s1.slow_logdel, h0 = s1.hb_stale, f0 = s1.fast_fail;
    e.get(1, K, "T3 (host1, after PUT#0 returned)");
    e.log.push_back("host1: installs refused " + std::to_string(r0) + "; T3 off the fast path because: no entry " +
                    std::to_string(s1.slow_noentry - n0) + ", deleted " + std::to_string(s1.slow_logdel - d0) +
                    ", heartbeat " + std::to_string(s1.hb_stale - h0) + ", check failed " + std::to_string(s1.fast_fail - f0));
    t1.join();
    disarm_all();
    ok &= !gA.timed_out && !gR.timed_out;
    if (!r.put_serial) ok &= p1;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// A1-asyncput (P2 with the asynchronous executor): backend writes take 200 us; a second thread keeps the owner's
// lanes busy with PUTs of 16 other keys of owner 0 (from both hosts). Host 1 PUTs K (owner 0) 30 times; right after each
// PUT returns, hosts 0 and 1 GET K and the checker requires that PUT's version or a newer one (P2). The owner never sent a
// reply before its due or its publish (reply_early = 0) and, with put_serial on, its PUTs went through the async executor.
static Result async_putreturn(const Rules &r) {
    g_put_delay_ns = 200000;
    Env e(r);
    g_put_delay_ns = 0;
    uint64_t K = key_owned_by(e, 0);
    std::vector<uint64_t> others;
    for (uint64_t a = K; others.size() < 16;) others.push_back(a = key_owned_by(e, 0, a));
    e.get(0, K, "preload");
    e.get(1, K, "preload");
    std::atomic<bool> done{false};
    std::atomic<uint64_t> bg{0};
    std::thread t([&] {
        for (uint64_t i = 0; !done.load(); i++) { e.put_quiet((int)(i % 2), others[i % others.size()]); bg++; }
    });
    for (int i = 0; i < 30; i++) {
        e.put(1, K, "PUT K (host1)");
        e.get(0, K, "GET right after (host0, owner)");
        e.get(1, K, "GET right after (host1)");
    }
    done = true;
    t.join();
    Stats &s0 = e.stats(0);
    e.log.push_back("background PUTs " + std::to_string(bg.load()) + "; owner: async PUTs " +
                    std::to_string(s0.put_async_puts.load()) + ", reply_early " + std::to_string(s0.reply_early.load()));
    bool ok = s0.reply_early.load() == 0;
    if (r.put_serial && r.log && !r.lease && !r.revoke) ok &= s0.put_async_puts.load() > 0;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// G2 (fails in log-hbafter): host 1's GET finds the old entry and loads it; then a poll deletes the entry and
// publishes a fresh heartbeat; then the GET checks the heartbeat. Read after the loads it is fresh and the old value is
// returned although the GET started after the PUT returned; read before the lookup it is older than P.
static Result log_hbafter(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    e.put(0, K, "T1 (returns W after its cut; host1 has not polled since)");
    Gate &gL = new_gate(Hook::FastAfterLoad, 1, K);
    arm(gL);
    std::thread t2([&] { e.get(1, K, "T2 (loads the old object, stalls before the heartbeat check)"); });
    ok &= wait_until([&] { return gL.reached.load(); });
    e.poll_once(1);
    e.log.push_back("host1 polls: applies K's record (deletes the entry), publishes a fresh heartbeat");
    gL.released = true;
    t2.join();
    disarm_all();
    ok &= !gL.timed_out;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// G3 (fails in log-noinstallcheck and log-snapafter): host 1's slow path reads the OLD AI entry and stalls;
// the whole PUT runs and host 1 applies its record (no entry to delete yet); then the slow path installs the old entry
// -- and every later hit (fresh heartbeat) returns the old value, forever. The check "no record for K applied since s0"
// (s0 taken before the AI read) refuses the install.
static Result log_installcheck(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.poll_once(1);
    Gate &gR = new_gate(Hook::SlowAfterAiRead, 1, K);
    arm(gR);
    std::thread t2([&] { e.get(1, K, "T2 (slow path reads the old AI entry, stalls across the PUT)"); });
    ok &= wait_until([&] { return gR.reached.load(); });
    e.put(0, K, "T1");
    e.poll_once(1);
    e.log.push_back("host1 polls: applies K's record (no entry to delete yet)");
    uint64_t ref0 = e.stats(1).log_refused;
    gR.released = true;
    t2.join();
    disarm_all();
    e.log.push_back("installs refused on host1: " + std::to_string(e.stats(1).log_refused - ref0));
    e.poll_once(1);
    e.get(1, K, "T3 (after PUT, fresh heartbeat)");
    ok &= !gR.timed_out;
    if (r.log_install_check && r.log_snap_before) ok &= e.stats(1).log_refused > ref0;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// G4 (fails in log-appendfirst): the PUT stalls between its two cut steps. Appending first, host 1 applies the
// record, then its slow path (s0 after the record) still reads the old AI entry and installs it; once the PUT
// unpublishes and returns, a hit returns the old value. Unpublishing first leaves nothing old to read.
static Result log_appendfirst(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.poll_once(1);
    Gate &gS = new_gate(Hook::PutAfterStep1, 0, K);
    arm(gS);
    std::thread t1([&] { e.put(0, K, "T1 (stalls between its two cut steps)"); });
    ok &= wait_until([&] { return gS.reached.load(); });
    e.poll_once(1);
    e.log.push_back("host1 polls during the PUT");
    e.get(1, K, "T2 (during the PUT)");
    gS.released = true;
    t1.join();
    disarm_all();
    e.poll_once(1);
    e.get(1, K, "T3 (after PUT, fresh heartbeat)");
    ok &= !gS.timed_out;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// G5 (fails in log-atpublish): a PUT that does not publish a new generation (here put_publish off; in general
// no space) must still append: host 1's entry for the old object would otherwise survive every poll.
static Result log_atpublish(const Rules &r0) {
    Rules r = r0;
    r.put_publish = false;
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.put(0, K, "T1 (not published: K leaves the cache)");
    e.poll_once(1);
    e.get(1, K, "T2 (after PUT, fresh heartbeat)");
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// G6 (fails in log-noclock, freshness form; the log removes lease L7's limitation): host 1's poller stalls; the PUT returns;
// host 1 only hits. The hit's heartbeat-age check sends it to the slow path; without the check it reads the old value.
static Result log_pollstall(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    e.log.push_back("host1's poller stalls");
    e.put(0, K, "T1");
    if (committed_rules(r)) {   // the committed ack replies at its cut; the bound is P after it
        nap(TP + TP / 20);
        e.log.push_back("(committed) P passes after the cut");
    }
    e.get(1, K, "T2 (after PUT, hit only)");
    if (r.log_hit_clock) ok &= e.last_kind != FAST;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// G7 (Rules::log_overrun: falling behind by more than the ring): ring of 2 lines = 14 records; host 1 holds K; 17 PUTs (K first)
// before host 1 polls. K's record was overwritten unread: the poll must drop all of owner 0's entries.
static Result log_overrun(const Rules &r) {
    Tweak tw([](Config &c) { c.log_lines = 2; c.log_poll_ns = 50'000'000; });
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    e.put(0, K, "T1");
    for (uint64_t k = K, i = 0; i < 16; i++) {
        k = key_owned_by(e, 0, k);
        e.put(0, k, "filler");
    }
    uint64_t ov0 = e.stats(1).log_overrun;
    e.poll_once(1);
    e.log.push_back("host1 polls: 17 records behind, the ring holds 14; overruns detected: " +
                    std::to_string(e.stats(1).log_overrun - ov0));
    e.get(1, K, "T2 (after PUT, fresh heartbeat)");
    if (r.log_overrun) ok &= e.stats(1).log_overrun > ov0;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// G8 (new owner, Rules::log_epoch): host 1 holds K and has consumed one record of owner 0's log; a new owner takes over
// region 0 (objects dropped without records, a new log from 0 in a new epoch, waits W; it continues the allocation
// counter -- had it restarted the counter from 0, the drop-all lag rule (log_dropall_lag) would also drop everything);
// its PUTs on K and Z append records #0 and #1. A reader that ignores the epoch resumes at #1 and keeps K's old entry.
static Result log_epoch(const Rules &r0) {
    Rules r = r0;
    r.put_publish = false;
    Tweak tw([](Config &c) { c.log_poll_ns = 50'000'000; });
    Env e(r);
    uint64_t K = key_owned_by(e, 0), A = key_owned_by(e, 0, K), Z = key_owned_by(e, 0, A);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.put(0, A, "PUT A (record #0 of the old epoch)");
    e.poll_once(1);
    e.owner_reset(0);
    e.log.push_back("owner region 0 taken over: objects dropped without records, new log epoch (counter continues)");
    e.put(0, K, "T1 (record #0 of the new epoch)");
    e.put(0, Z, "PUT Z (record #1)");
    uint64_t ec0 = e.stats(1).log_epoch_chg;
    e.poll_once(1);
    e.get(1, K, "T2 (after PUT, fresh heartbeat)");
    if (r.log_epoch) ok &= e.stats(1).log_epoch_chg > ec0;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// ---- generation wrap with gen_bits = 6 (64 generations, 0 skipped): restamp at age 16,
// sweep at age 32, drop-all when the counter moved >= 32 between examinations. They need stale lines in host 1's cache:
// software non-coherent cache. X and Y are two-line objects of owner 0 in the same size class, so Y reuses X's slot.
constexpr int WRAP_BITS = 6;
static uint64_t gen_dist(uint32_t from, uint32_t to) {   // allocations until the next generation is `to`
    const uint32_t m = (1u << WRAP_BITS) - 1;
    uint64_t n = 0;
    for (uint32_t c = from; c != to; n++) { c = (c + 1) & m; if (!c) c = 1; }
    return n;
}
// advance owner 0's counter until its next generation is g; with step > 0 host 1 polls after every `step` allocations
static void advance_to(Env &e, uint32_t g, uint64_t step, bool poll) {
    for (;;) {
        uint64_t d = gen_dist(e.owner_next_gen(0), g);
        if (!d) break;
        e.owner_advance(0, step ? std::min(d, step) : d);
        if (poll) e.poll_once(1);
    }
}
// W1 / W2 (fail in log-nosweep / log-nodropall): host 1 installs X and keeps line 0 cached; X is evicted (no record);
// the counter goes a full round; Y is admitted into X's slot with X's generation. Host 1's GET of X then takes line 0
// from its cache (X, generation g) and line 1 from memory (Y's, generation g): a torn object. W1 polls every 8
// allocations (the sweep drops X's entry at age >= 32); W2 polls only after the round (the counter moved >= 32 since
// the last examination: drop all; the age mod 64 alone looks young).
static Result wrap_reuse(const Rules &r, bool poll_during) {
    Tweak tw([](Config &c) { c.gen_bits = WRAP_BITS; });
    Env e(r, true);
    uint64_t X = key_two_lines(e, 0), Y = key_two_lines(e, 0, X);
    e.pause_poller(1, true);
    e.get(0, X, "preload X");
    bool ok = wait_until([&] { return e.owner_cached(X); });
    uint32_t gX = e.owner_gen(X);
    int64_t off = e.owner_offset(X);
    e.poll_once(1);
    e.get(1, X, "T0 (host1 installs X, caches both lines)");
    e.emu_drop_line(1, 0, (uint64_t)off + 64);
    ok &= e.owner_evict(X);
    e.log.push_back("host1's cache keeps line 0 of X only; owner evicts X (no record); gen(X) = " + std::to_string(gX));
    advance_to(e, gX, poll_during ? 8 : 0, poll_during);
    e.log.push_back(std::string("owner's counter goes a full round") + (poll_during ? ", host1 polls every 8" : ", host1 does not poll"));
    e.get(0, Y, "owner host GET Y (admitted into X's slot)");
    ok &= wait_until([&] { return e.owner_cached(Y); }) && e.owner_offset(Y) == off && e.owner_gen(Y) == gX;
    e.log.push_back("Y: slot " + std::to_string(e.owner_offset(Y)) + " gen " + std::to_string(e.owner_gen(Y)));
    e.poll_once(1);
    e.get(1, X, "T1 (host1 GET X, fresh heartbeat)");
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}
static Result wrap_sweep(const Rules &r) { return wrap_reuse(r, true); }
static Result wrap_dropall(const Rules &r) { return wrap_reuse(r, false); }

// W3 (fails in log-norestamp and lease-norestamp): X stays cached while the counter goes
// almost a full round; host 1 then installs X fresh (slow path), keeps line 0; X is evicted and Y takes the slot with the
// next generation, X's original one. Without restamp X still has it: host 1's hit is torn. With restamp X's generation
// is never more than 16 old.
static Result wrap_restamp(const Rules &r) {
    Tweak tw([](Config &c) { c.gen_bits = WRAP_BITS; });
    Env e(r, true);
    uint64_t X = key_two_lines(e, 0), Y = key_two_lines(e, 0, X);
    if (r.log) e.pause_poller(1, true);
    e.get(0, X, "preload X");
    bool ok = wait_until([&] { return e.owner_cached(X); });
    uint32_t gX = e.owner_gen(X);
    int64_t off = e.owner_offset(X);
    advance_to(e, gX, 8, r.log);
    e.log.push_back("owner's counter goes almost a full round with X cached; gen(X) now " + std::to_string(e.owner_gen(X)) +
                    " (was " + std::to_string(gX) + "), restamps " + std::to_string(e.stats(0).restamps));
    if (r.log) e.poll_once(1);
    e.get(1, X, "T0 (host1 installs X, caches both lines)");
    e.emu_drop_line(1, 0, (uint64_t)off + 64);
    ok &= e.owner_evict(X);
    e.get(0, Y, "owner host GET Y (admitted into X's slot)");
    ok &= wait_until([&] { return e.owner_cached(Y); }) && e.owner_offset(Y) == off;
    e.log.push_back("host1 keeps line 0 of X; owner evicts X; Y: slot " + std::to_string(e.owner_offset(Y)) + " gen " +
                    std::to_string(e.owner_gen(Y)));
    e.get(1, X, "T1 (host1 GET X)");
    if (r.log) e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// W4 (fails in log-noclock): host 1's hit reads its entry for X and stalls (longer than P) while X is
// evicted, the counter goes a full round and Y takes the slot with X's generation; resumed, it loads line 0 from its cache
// and line 1 from memory. Only the heartbeat-age check after the loads catches the pause.
static Result wrap_midstall(const Rules &r) {
    Tweak tw([](Config &c) { c.gen_bits = WRAP_BITS; });
    Env e(r, true);
    uint64_t X = key_two_lines(e, 0), Y = key_two_lines(e, 0, X);
    e.pause_poller(1, true);
    e.get(0, X, "preload X");
    bool ok = wait_until([&] { return e.owner_cached(X); });
    uint32_t gX = e.owner_gen(X);
    int64_t off = e.owner_offset(X);
    e.poll_once(1);
    e.get(1, X, "T0 (host1 installs X, caches both lines)");
    e.emu_drop_line(1, 0, (uint64_t)off + 64);
    e.poll_once(1);
    Gate &gF = new_gate(Hook::FastAfterLiRead, 1, X);
    arm(gF);
    std::thread t1([&] { e.get(1, X, "T1 (reads its entry for X, stalls longer than P)"); });
    ok &= wait_until([&] { return gF.reached.load(); });
    ok &= e.owner_evict(X);
    advance_to(e, gX, 8, true);
    e.get(0, Y, "owner host GET Y (admitted into X's slot)");
    ok &= wait_until([&] { return e.owner_cached(Y); }) && e.owner_offset(Y) == off && e.owner_gen(Y) == gX;
    e.log.push_back("during the stall: X evicted, a full round, Y in X's slot with gen " + std::to_string(e.owner_gen(Y)));
    nap(TP + 50'000'000);
    gF.released = true;
    t1.join();
    disarm_all();
    ok &= !gF.timed_out;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// ================================================================ change-feed scenarios (Rules::feed, kg in CXL)
// F3 (fails in kg-late: P3): host 1 misses on K (not cached), reads the OLD backend value and stalls before
// sending its fill; the whole PUT runs (both kg bumps written back) and returns without publishing; then the fill goes
// out. With the token loaded before the backend read it is stale and the owner drops the fill; loaded after, it equals
// the owner's word and the old value is admitted after the PUT returned (P3).
static Result feed_kglate(const Rules &r0) {
    Rules r = r0;
    r.put_publish = false;
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    ok &= e.owner_evict(K);
    e.log.push_back("owner evicts K");
    Gate &gF = new_gate(Hook::MissBeforeSendFill, 1, K);
    arm(gF);
    std::thread t1([&] { e.get(1, K, "T1 (miss: reads the old backend value, stalls before sending its fill)"); });
    ok &= wait_until([&] { return gF.reached.load(); });
    e.put(0, K, "T2 (PUT, not published, returns)");
    gF.released = true;
    t1.join();
    disarm_all();
    short_settle();
    e.log.push_back(drops(e, 0));
    e.get(1, K, "T3 (after PUT)");
    ok &= !gF.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// F4 (fails in refresh-noflush; gen_bits = 6): host 1 installs X and keeps line 1 of generation g in its
// cache; X stays cached while the counter goes almost a round (restamps keep it young and refresh host 1's entry); X is
// evicted and a PUT puts X's new value into the same slot with generation g again, appending delete + refresh records;
// host 1's poller repoints its entry to (slot, g). Without flushing first, line 0 comes from memory (new value) and line 1
// from the cache (old value): a torn object.
static Result feed_refreshflush(const Rules &r) {
    Tweak tw([](Config &c) { c.gen_bits = WRAP_BITS; });
    Env e(r, true);
    uint64_t X = key_two_lines(e, 0, 0, true);
    e.pause_poller(1, true);
    e.get(0, X, "preload X");
    bool ok = wait_until([&] { return e.owner_cached(X); });
    uint32_t gX = e.owner_gen(X);
    int64_t off = e.owner_offset(X);
    e.poll_once(1);
    e.get(1, X, "T0 (host1 installs X, caches both lines)");
    e.emu_drop_line(1, 0, (uint64_t)off);
    e.log.push_back("host1's cache keeps line 1 of X (gen " + std::to_string(gX) + ") only");
    advance_to(e, gX, 8, true);
    e.log.push_back("counter almost a full round with X cached; restamps " + std::to_string(e.stats(0).restamps) +
                    ", host1 refreshes " + std::to_string(e.stats(1).log_refreshed));
    ok &= e.owner_evict(X);
    advance_to(e, gX, 1, false);
    std::string tag = "T1 (PUT X: new value into X's old slot with gen " + std::to_string(gX) + ")";
    std::thread t1([&] { e.put(0, X, tag.c_str()); });
    ok &= wait_until([&] { return e.owner_cached(X) && e.owner_offset(X) == off; });
    e.log.push_back("X v1 published: slot " + std::to_string(e.owner_offset(X)) + " gen " + std::to_string(e.owner_gen(X)));
    e.poll_once(1);
    e.log.push_back("host1 polls: delete + refresh records for X");
    e.get(1, X, "T2 (host1 GET X)");
    t1.join();
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// F5 (keep_failed): host 1 holds X; X is evicted and its slot reused by Y, so host 1's next read
// fails the check -- the entry stays held, marked invalid; X is re-admitted (refresh record) and host 1 polls: its entry
// is repointed (lines flushed first), so its next read is a fast hit. Erasing the entry instead (feed-erasefailed) means the
// refresh finds nothing to repoint and the read takes the slow path. Expectation, not a counterexample: no errors either way.
static Result feed_keepfailed(const Rules &r) {
    Env e(r);
    uint64_t X = key_two_lines(e, 0), Y = key_two_lines(e, 0, X);
    e.pause_poller(1, true);
    e.get(0, X, "preload X");
    bool ok = wait_until([&] { return e.owner_cached(X); });
    int64_t off = e.owner_offset(X);
    e.poll_once(1);
    e.get(1, X, "T0 (host1 installs X)");
    ok &= e.owner_evict(X);
    e.get(0, Y, "owner host GET Y (takes X's slot)");
    ok &= wait_until([&] { return e.owner_cached(Y); }) && e.owner_offset(Y) == off;
    e.poll_once(1);
    e.get(1, X, "T1 (host1: check fails, miss)");
    e.get(0, X, "owner host GET X (re-admitted)");
    ok &= wait_until([&] { return e.owner_cached(X); });
    e.poll_once(1);
    e.log.push_back("host1 polls: refresh record for X; refreshed so far " + std::to_string(e.stats(1).log_refreshed));
    e.get(1, X, "T2 (host1 GET X)");
    if (r.keep_failed) ok &= e.last_kind == FAST;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// ================================================================ swap order / committed ack (Rules::put_cut_at_publish)

// L3 under the swap order (fails in *-swap-nofilltok): K is not cached, so host 1 misses: it takes its fill token
// while the PUT is in its backend write (the swap order bumps kg only at the swap) and reads the old value; the PUT swaps,
// replies, and its new object is evicted; then the fill arrives. The token check (the swap's kg bump) rejects it.
static Result swap_stalefill(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    ok &= e.owner_evict(K);
    e.log.push_back("owner evicts K (host 1 will miss)");
    Gate &gP = new_gate(Hook::PutAfterAcks, 0, K), &gF = new_gate(Hook::MissBeforeSendFill, 1, K);
    arm(gP);
    std::thread t1([&] { e.put(0, K, "T1"); });
    ok &= wait_until([&] { return gP.reached.load(); });   // pending, backend write not issued yet
    arm(gF);
    std::thread t2([&] { e.get(1, K, "T2 (miss during the PUT: token, then the old backend value)"); });
    ok &= wait_until([&] { return gF.reached.load(); });
    gP.released = true;
    t1.join();
    ok &= wait_until([&] { return e.owner_cached(K); }) && e.owner_evict(K);
    e.log.push_back("owner evicts K (the PUT's new object)");
    gF.released = true;
    t2.join();
    short_settle();
    e.log.push_back(drops(e, 0));
    nap(TP + TP / 20);
    e.get(1, K, "T3 (P after the PUT's cut)");
    disarm_all();
    ok &= !gP.timed_out && !gF.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// C1 (P4 for the committed ack): host 1 holds a fresh entry and heartbeat and its poller stalls; the PUT replies
// (committed: at its cut); a GET right after may return the old value (no RAW); one that starts P after the cut must not:
// the hit's heartbeat-age check sends it to the slow path, which reads the switched AI entry. Without that check
// (log_hit_clock off) the hit returns the old value forever. Then host 1 polls; the next GET is checked again.
static Result committed_boundedstale(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    e.log.push_back("host1 polls (fresh heartbeat); its poller stalls");
    e.put(0, K, "T1");
    e.get(1, K, "T2 (right after the PUT replied)");
    // the committed reply came before any poll: a stale hit is legal (with Config::mono: K is in host 1's table since
    // T0, so AI names another object now and the GET reads the backend instead)
    if (committed_rules(r) && !e.cfg.mono) ok &= e.last_kind == FAST;
    nap(TP + TP / 20);
    e.get(1, K, "T3 (P after the cut, poller still stalled)");
    e.poll_once(1);
    e.get(1, K, "T4 (after a poll)");
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// C2 (fails in the *-delfirst variants): the PUT stops inside the swap, after its first step (design: the switch to the new
// object; ablation swap_del_first: the delete record = the cut); host 1, which holds K, polls (the delete record, if any, is
// applied), P passes, and host 1 GETs: no valid entry or a stale heartbeat -> slow path. Design: AI already names the new
// object. Ablation: AI still names the old one, more than P after the cut -> stale read (and installed).
static Result swap_order(const Rules &r) {
    if (!r.put_cut_at_publish) {
        Result x{0, true, {"n/a (window order: no swap)"}, {}};
        return x;
    }
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    Gate &gS = new_gate(Hook::PutSwapMid, 0, K);
    arm(gS);
    std::thread t1([&] { e.put(0, K, "T1"); });
    ok &= wait_until([&] { return gS.reached.load(); });
    e.log.push_back(r.swap_del_first ? "PUT stopped after its delete record (cut), before the switch"
                                     : "PUT stopped after the switch, before its delete record (cut)");
    e.poll_once(1);
    e.log.push_back("host1 polls");
    nap(TP + TP / 20);
    e.get(1, K, "T2 (P later, PUT still inside the swap)");
    gS.released = true;
    t1.join();
    e.poll_once(1);
    e.get(1, K, "T3 (after the PUT, a poll)");
    nap(TP + TP / 20);
    e.get(1, K, "T4 (P later)");
    disarm_all();
    e.pause_poller(1, false);
    ok &= !gS.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// X2 (XTRA-style own freshness): host 1 holds an entry for K's object; the owner evicts K (no message, no record); a PUT on
// K installs the new version in another size class (the old slot is not reused); host 1 GETs after the PUT returned. Its
// entry must fail the directory comparison: the owner zeroed the old object's word at the eviction (xtra_kill_on_evict).
static Result xtra_evictput(const Rules &r) {
    Env e(r);
    uint64_t K = 0;
    for (uint64_t k = 1; k < 1024 && !K; k++) {   // v0 and v1 of different line counts <= 4 lines: different size classes
        uint32_t a0 = obj_lines(value_len(k, 0)), a1 = obj_lines(value_len(k, 1));
        if (e.owner_of(k) == 0 && a0 != a1 && a0 <= 4 && a1 <= 4) K = k;
    }
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = K && wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.get(1, K, "T0b (host1 hits K)");
    ok &= e.last_kind == FAST;
    ok &= e.owner_evict(K);
    e.log.push_back("owner evicts K (no message, no record)");
    e.put(0, K, "T1 (PUT: K not cached, the new object goes to another size class)");
    e.get(1, K, "T2 (after the PUT returned)");
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// X4: the local index's CLOCK queue under XTRA-style revalidation. Host 1 holds K; 40 times a PUT on K (host 0) moves K's
// slot version and host 1's next GET of K fails its hit, erases the entry and the slow path reinstalls it. With
// Config::li_clock_inq (default) the reinstall finds K's slot already queued: the queue stays at a few entries and no
// dedup pass runs; with li_clock_inq = false every reinstall pushes again (>= 40 entries; the dedup pass, O(queue) under
// poll_mu, then runs once the queue passes 2 x entries + 4096).
static Result xtra_clockq(const Rules &r) {
    Result res{0, true, {}, {}};
    for (int inq = 1; inq >= 0; inq--) {
        Tweak tw([inq](Config &c) { c.li_clock_inq = inq != 0; });
        Env e(r);
        uint64_t K = key_owned_by(e, 0);
        e.get(0, K, "preload");
        bool ok = wait_until([&] { return e.owner_cached(K); });
        e.get(1, K, "T0 (host1 installs K)");
        uint64_t slow0 = e.stats(1).get_slow;
        for (int i = 0; i < 40; i++) {
            e.put(0, K, "put");
            e.get(1, K, "revalidate");
        }
        e.poll_once(1);   // a housekeeping round: li_cap updates the queue gauge
        const uint64_t q = e.stats(1).li_clock_q, d = e.stats(1).li_clock_dedups, slow = e.stats(1).get_slow - slow0;
        e.log.push_back(std::string("li_clock_inq=") + std::to_string(inq) + ": slow-path GETs " + std::to_string(slow) +
                        ", clock queue " + std::to_string(q) + ", dedup passes " + std::to_string(d));
        ok &= slow >= 30;   // the revalidations took the slow path (a PUT moves the slot version)
        ok &= inq ? q <= 4 && d == 0 : q >= 30;
        res.errors += e.errors();
        res.reproduced &= ok;
        for (auto &l : std::vector<std::string>(e.log)) res.log.push_back(l);
        for (auto &l : e.first_errors()) res.errs.push_back(l);
    }
    return res;
}

// X3 (XTRA's Coordination Directory version semantics): every key of an owner shares ONE directory slot (xtra_dir_slots =
// 2: one per owner); host 1 holds entries for K and J; a PUT on J moves the shared slot's version; K's next hit fails once
// (the slow path reinstalls it under the new version) and K hits again afterwards. With a generation comparison instead
// (the word holding the last published object's generation) K would miss the fast path on every GET from then on.
static Result xtra_collide(const Rules &r) {
    // Copies need at least one complete line each. Retain the one-slot test when copies are off;
    // otherwise choose two owner-0 keys sharing one of its eight slots (same X3 collision).
    Tweak tw([](Config &c) { c.xtra_dir_slots = g_xrep ? 16 : 2; });
    Env e(r);
    uint64_t K = key_owned_by(e, 0), J = key_owned_by(e, 0, K);
    if (g_xrep) {
        auto slot = [](uint64_t key) {
            // xslot's mix64(key ^ salt); splitmix adds salt before applying the same mixer.
            constexpr uint64_t salt = 0x9e3779b97f4a7c15ULL;
            uint64_t state = (key ^ salt) - salt;
            return splitmix(state) & 7;
        };
        while (slot(K) != slot(J)) J = key_owned_by(e, 0, J);
    }
    e.get(0, K, "preload K");
    e.get(0, J, "preload J");
    bool ok = wait_until([&] { return e.owner_cached(K) && e.owner_cached(J); });
    e.get(1, K, "T0 (host1 installs K)");
    e.get(1, J, "T0b (host1 installs J)");
    e.get(1, K, "T1 (K hits)");
    ok &= e.last_kind == FAST;
    e.put(0, J, "T2 (PUT J: the shared slot's version moves)");
    e.get(1, K, "T3 (K: version moved -> one revalidation)");
    ok &= e.last_kind == SLOW;
    e.get(1, K, "T4 (K hits again)");
    ok &= e.last_kind == FAST;
    e.get(1, J, "T5 (J: new version)");
    e.get(1, J, "T6 (J hits)");
    ok &= e.last_kind == FAST;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// ================================================================ DELETE (Client::del; controls: the del-nounpub /
// del-nobump / del-norec variants and swap_del_first). A DELETE takes its PUT's path and ack level:
// window order and swap + W are RAW (a GET that starts after the reply misses: absent), swap without W is committed (a GET
// that starts P after the cut must not return the old value). The checker takes the DELETE's version (the backend
// tombstone) as a completed write: an older value afterwards is P2 (RAW) / STALE (committed); an absent result is checked
// against the backend. Host 1's poller is paused and its polls forced, as in the log suite (no-ops without the log).
// Only the setup is asserted ("forced"); every outcome is judged by the checker.

// D1 (G1 for a DELETE, P2-P4; controls del-nounpub, del-norec): host 1 holds a fresh entry for K and a fresh heartbeat; the
// DELETE replies; host 1 GETs right after (RAW: absent), after a poll (any variant: the delete record dropped the entry),
// and P after the cut with a fresh poll (committed). Then the owner host, a PUT, and P later its value. norec: host 1's
// entry survives every poll and keeps hitting the old object; nounpub: the slow path still finds the old AI entry;
// XTRA's own freshness without the slot bump (nokillput): the entry keeps passing its version check.
static Result del_raw(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = g_scheme == Scheme::PRIV || g_scheme == Scheme::SHARD || wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 caches / installs K)");
    e.get(1, K, "T0b (host1 hits K)");
    ok &= e.last_kind != MISS;
    e.poll_once(1);
    e.log.push_back("host1 polls (fresh heartbeat); its poller stays paused");
    ok &= e.del(0, K, "T1 (DELETE)") != 0;
    e.get(1, K, "T2 (right after the DELETE returned)");
    e.poll_once(1);
    e.get(1, K, "T3 (after a poll)");
    nap(TP + TP / 20);
    e.poll_once(1);
    e.get(1, K, "T4 (P after the cut, fresh poll)");
    e.get(0, K, "T5 (owner host)");
    uint32_t v = e.put(1, K, "T6 (PUT after the DELETE)");
    ok &= v != 0;
    nap(TP + TP / 20);
    e.poll_once(1);
    e.get(1, K, "T7 (P after the PUT)");
    e.get(1, K, "T8");
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// D2 (control del-nobump: P3, a fill read before the DELETE is admitted after it): K is not cached; host 1 misses, takes its fill
// token and reads the backend's value, and its fill is held; the whole DELETE runs; then the fill arrives. The DELETE's kg
// bump (window: at its start and bump2; swap: at the switch) rejects it; without the bump it is admitted (P3) and host 1
// then reads the deleted value (P2 / STALE). PRIV / SHARD: the invalidation round bumps the private kg.
static Result del_stalefill(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    Gate &gF = new_gate(Hook::MissBeforeSendFill, 1, K);
    arm(gF);
    std::thread t1([&] { e.get(1, K, "T1 (miss before the DELETE: token, then the backend value; the fill is held)"); });
    bool ok = wait_until([&] { return gF.reached.load(); });
    ok &= e.del(0, K, "T2 (DELETE while host1's fill is held)") != 0;
    gF.released = true;
    t1.join();
    short_settle();
    e.log.push_back(drops(e, 0) + " | " + drops(e, 1));
    nap(TP + TP / 20);   // committed: past the bounded-staleness deadline
    e.poll_once(1);
    e.get(1, K, "T3 (after the DELETE)");
    e.get(1, K, "T3b");
    e.get(0, K, "T4 (owner host)");
    disarm_all();
    ok &= !gF.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// D3 (TTL): EXPIRE = the DELETE's protocol without the backend write: both hosts hold K; the owner expires it; host 1's next
// GET (RAW) misses with the SAME version (not absent) and refills; later GETs hit again. No version moves, so the checker
// cannot fail here: the scenario asserts the miss (RAW variants), the version, the refill and the owner's expire count.
static Result del_expire(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    uint32_t v0 = e.last_ver;
    bool ok = g_scheme == Scheme::PRIV || g_scheme == Scheme::SHARD || wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 caches / installs K)");
    e.poll_once(1);
    ok &= e.expire(0, K, "T1 (EXPIRE)") != 0;
    e.get(1, K, "T2 (right after the EXPIRE returned)");
    if (!committed_rules(r)) ok &= e.last_kind == MISS;
    ok &= e.last_ver == v0 && !e.last_absent;
    e.poll_once(1);
    short_settle();
    e.get(1, K, "T3 (after the refill; committed: the miss that refills)");
    short_settle();
    e.get(1, K, "T4");
    ok &= e.last_kind != MISS && e.last_ver == v0;
    ok &= e.stats(0).expires == 1 && e.stats(0).deletes == 0;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// D4: DELETE from the non-owner host, DELETE again (a new tombstone version), PUT after the DELETEs; versions move up, GETs on
// both hosts see absent / the new value, the owner counts two deletes.
static Result del_delput(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    e.get(1, K, "preload");
    uint32_t v1 = e.del(1, K, "T1 (DELETE from host1)");
    e.poll_once(1);
    e.get(0, K, "T2");
    e.get(1, K, "T2b");
    uint32_t v2 = e.del(0, K, "T3 (DELETE again)");
    nap(TP + TP / 20);
    e.poll_once(1);
    e.get(1, K, "T4");
    bool ok = v1 && v2 > v1 && e.last_absent && e.last_ver == v2;
    uint32_t v3 = e.put(1, K, "T5 (PUT)");
    ok &= v3 > v2;
    nap(TP + TP / 20);
    e.poll_once(1);
    e.get(1, K, "T6");
    ok &= !e.last_absent && e.last_ver == v3;
    e.get(1, K, "T7");
    e.get(0, K, "T8");
    ok &= !e.last_absent && e.last_ver == v3;
    ok &= e.stats(0).deletes == 2;
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// D5 (C2 for a DELETE; control swap_del_first: the delete record before the removal, P4): the DELETE stops inside the
// swap, after its first step (design: AI := None; swap_del_first: the delete record = the cut); host 1, which holds K,
// polls, P passes, host 1 GETs. Design: the AI entry is gone -> absent. Ablation: the record dropped host 1's entry but
// the AI entry still names the old object more than P after the cut -> stale read.
static Result del_swapmid(const Rules &r) {
    if (!r.put_cut_at_publish) {
        Result x{0, true, {"n/a (window order: no swap)"}, {}};
        return x;
    }
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    Gate &gS = new_gate(Hook::PutSwapMid, 0, K);
    arm(gS);
    std::thread t1([&] { e.del(0, K, "T1 (DELETE)"); });
    ok &= wait_until([&] { return gS.reached.load(); });
    e.log.push_back(r.swap_del_first ? "DELETE stopped after its delete record (cut), before removing the object"
                                     : "DELETE stopped after removing the object (AI := None), before its delete record (cut)");
    e.poll_once(1);
    e.log.push_back("host1 polls");
    nap(TP + TP / 20);
    e.get(1, K, "T2 (P later, DELETE still inside the swap)");
    gS.released = true;
    t1.join();
    e.poll_once(1);
    e.get(1, K, "T3 (after the DELETE, a poll)");
    nap(TP + TP / 20);
    e.get(1, K, "T4 (P later)");
    disarm_all();
    e.pause_poller(1, false);
    ok &= !gS.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// D6 (C1 for a DELETE, committed; control *-noclock): host 1 holds a fresh entry and heartbeat and its poller stalls; the
// DELETE replies (committed: at its cut); a GET right after may hit the deleted value; one that starts P after the cut must
// not (the heartbeat-age check sends it to the slow path: absent). Without the check it hits the old object forever.
static Result del_boundedstale(const Rules &r) {
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    e.log.push_back("host1 polls (fresh heartbeat); its poller stalls");
    ok &= e.del(0, K, "T1 (DELETE)") != 0;
    e.get(1, K, "T2 (right after the DELETE replied)");
    // the committed reply came before any poll: a stale hit is legal (VER has no local entry: it reads the AI entry every GET)
    if (committed_rules(r) && g_scheme != Scheme::VER && g_scheme != Scheme::VER_NOFLUSH) ok &= e.last_kind == FAST;
    nap(TP + TP / 20);
    e.get(1, K, "T3 (P after the cut, poller still stalled)");
    e.poll_once(1);
    e.get(1, K, "T4 (after a poll)");
    e.pause_poller(1, false);
    return {e.errors(), ok, e.log, e.first_errors()};
}

// ================================================================ other owners' blocks read-only (Config::ro_peers)
// A forked child maps a fresh shared area as host 1 of 2 (shm, local_host = 1; no start(): the mapping is all it needs) and
// accesses owner 0's block; before each access it writes a byte to a pipe, so the parent knows how far it got; the parent
// checks how it ended (waitpid). Runs the same with or without --procs (it always forks).
struct ForkEnd { int steps; bool sig; int signo, code; };
static ForkEnd run_forked(Config c, const std::vector<std::array<int, 3>> &acc /* owner, kind, where */) {
    static int seq = 0;
    c.shm_name = "/rcro-" + std::to_string(getpid()) + "-" + std::to_string(seq++);
    c.local_host = 1;
    Cache::shm_create(c);
    int pfd[2];
    if (pipe(pfd)) { perror("pipe"); exit(1); }
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        close(pfd[0]);
        signal(SIGSEGV, SIG_DFL);
        SimBackend b(NKEYS, 0);
        Cache *ca = new Cache(c, &b);   // never destroyed: _exit below
        for (auto &a : acc) {
            char x = 1;
            if (write(pfd[1], &x, 1) != 1) _exit(3);
            ca->peer_access(a[0], a[1], a[2]);
        }
        _exit(0);
    }
    close(pfd[1]);
    int steps = 0;
    char x;
    while (read(pfd[0], &x, 1) == 1) steps++;
    close(pfd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    Cache::shm_unlink(c.shm_name);
    return {steps, WIFSIGNALED(st), WIFSIGNALED(st) ? WTERMSIG(st) : 0, WIFEXITED(st) ? WEXITSTATUS(st) : -1};
}
static const char *WHERE_NAMES[] = {"slab", "AI table", "outgoing rings", "consumer tails", "log tail", "kg words", "last line"};
static std::string fork_end_str(const ForkEnd &f) {
    return f.sig ? "killed by signal " + std::to_string(f.signo) + (f.signo == SIGSEGV ? " (SIGSEGV)" : "") + " at step " + std::to_string(f.steps)
                 : "exited " + std::to_string(f.code) + " after " + std::to_string(f.steps) + " steps";
}
// R1: in every part of owner 0's block, a load, clflush, clflushopt and clwb must pass and the store must fault (SIGSEGV at
// step 5). "errors" counts the parts where the store went through (control rw-peers: every part) or something else faulted.
static Result ro_peerwrite(const Rules &r) {
    Config c = test_config(r, false, 60'000'000'000ULL);
    Result res{0, true, {}, {}};
    for (int w = 0; w <= 6; w++) {
        std::vector<std::array<int, 3>> acc;
        for (int k = 0; k <= 4; k++) acc.push_back({0, k, w});
        ForkEnd f = run_forked(c, acc);
        const bool ok = f.sig && f.signo == SIGSEGV && f.steps == 5;
        res.log.push_back(std::string("host1 -> owner0 ") + WHERE_NAMES[w] + ": load, clflush, clflushopt, clwb, store: " +
                          fork_end_str(f) + (ok ? "" : "  <-- not the expected SIGSEGV at the store"));
        if (!ok) res.errors++;
        if (f.sig && f.steps < 5) res.reproduced = false;   // a non-store access faulted: the readers' flushes would too
    }
    return res;
}
// R2: the child's own block stays writable, and every non-store access to the peer's block passes: exit 0 after all steps.
static Result ro_ownwrite(const Rules &r) {
    Config c = test_config(r, false, 60'000'000'000ULL);
    std::vector<std::array<int, 3>> acc;
    for (int w = 0; w <= 6; w++) acc.push_back({1, 4, w});   // own block: stores
    for (int w = 0; w <= 6; w++)
        for (int k = 0; k <= 3; k++) acc.push_back({0, k, w});   // peer block: load / flushes / clwb
    ForkEnd f = run_forked(c, acc);
    const bool ok = !f.sig && f.code == 0 && f.steps == (int)acc.size();
    Result res{ok ? 0u : 1u, true, {"host1: stores into its own block (7 parts), then load / clflush / clflushopt / clwb in every part of "
                                     "owner0's block: " + fork_end_str(f) + " of " + std::to_string(acc.size())}, {}};
    return res;
}

// ================================================================ monotonic reads (P6) per host, Config::mono
// M1-ghost: the owner evicts K (no record), so host 1's entry names a ghost slot; a swap-order PUT is held after its
// backend write; host 1's heartbeat goes stale, its GET leaves the fast path, finds AI empty, misses and returns the NEW
// value from the backend; host 1 polls (fresh heartbeat, the PUT has not cut yet) and GETs again: the ghost entry still
// loads the OLD value. Without Config::mono that is a P6 violation (the RAW / committed checks do not object: the PUT has
// not completed). With it, the first GET recorded (K, kg loaded after the backend read); the second finds K in the
// table, reads AI[K] (empty: evicted) and takes the backend. The ablation mono_hold off (entries expire at once)
// violates again.
static Result mono_ghost(const Rules &r) {
    if (!r.put_cut_at_publish) {
        Result x{0, true, {"n/a (window order: the PUT's delete record at its start removes the ghost entry at the next poll)"}, {}};
        return x;
    }
    Env e(r);
    uint64_t K = key_owned_by(e, 0);
    e.pause_poller(1, true);
    e.get(0, K, "preload");
    bool ok = wait_until([&] { return e.owner_cached(K); });
    e.get(1, K, "T0 (host1 installs K)");
    e.poll_once(1);
    e.get(1, K, "T0b (host1 hits K)");
    ok &= e.last_kind == FAST;
    const uint32_t v0 = e.last_ver;
    ok &= e.owner_evict(K);
    e.log.push_back("owner evicts K (no record): host1's entry now names a ghost slot");
    Gate &gP = new_gate(Hook::PutAfterBackend, 0, K);
    arm(gP);
    std::thread t1([&] { e.put(0, K, "T1 (PUT, held after its backend write; pending, no cut yet)"); });
    ok &= wait_until([&] { return gP.reached.load(); });
    nap(TP + TP / 20);
    e.get(1, K, "T2 (stale heartbeat -> slow path, AI empty -> miss: the backend's new value)");
    ok &= e.last_kind == MISS && e.last_ver > v0;
    e.poll_once(1);
    e.log.push_back("host1 polls (fresh heartbeat; the held PUT has no record yet)");
    e.get(1, K, "T3 (fresh heartbeat; host1's ghost entry)");
    gP.released = true;
    t1.join();
    e.poll_once(1);
    e.get(1, K, "T4 (after the PUT)");
    disarm_all();
    e.pause_poller(1, false);
    ok &= !gP.timed_out;
    return {e.errors(), ok, e.log, e.first_errors()};
}

// Basic paths: miss -> admitted -> fast hit on the owner, slow then fast on a non-owner, PUT visible everywhere,
// eviction under a tiny budget, reuse of evicted slots detected by gen.
static Result basic(const Rules &r) {
    Env e(r);
    bool ok = true;
    for (int o = 0; o < 2; o++) {
        uint64_t K = key_owned_by(e, o);
        e.get(o, K, "b1");
        ok &= wait_until([&] { return e.owner_cached(K); });
        e.get(o, K, "b2");
        e.get(1 - o, K, "b3");
        e.get(1 - o, K, "b4");
        e.put(1 - o, K, "b5");
        e.get(0, K, "b6");
        e.get(1, K, "b7");
        e.get(1, K, "b8");
    }
    Stats &s0 = e.stats(0), &s1 = e.stats(1);
    // Config::mono without put-wait: until host 1 polls the swap's record, its entry names the replaced copy, so b7 / b8
    // of the second round may both read the backend instead of hitting (the table has K since b3)
    ok &= s0.get_fast + s1.get_fast >= (e.cfg.mono ? 3u : 4u) && s0.get_slow + s1.get_slow >= 2 && s0.get_miss + s1.get_miss >= 2;
    // churn every key through a tiny budget: evictions and slot reuse
    for (int round = 0; round < 3; round++)
        for (uint64_t k = 1; k < 64; k++) {
            e.get((int)(k % 2), k, "churn");
            if (k % 7 == 0) e.put((int)((k + 1) % 2), k, "churn-put");
        }
    e.log.push_back("evictions=" + std::to_string(s0.evictions + s1.evictions) +
                    " fast_fail=" + std::to_string(s0.fast_fail + s1.fast_fail));
    return {e.errors(), ok, e.log, e.first_errors()};
}

struct Variant { const char *name; Rules r; };
struct Scenario { const char *name; Result (*fn)(const Rules &); const char *must_fail_in; };   // comma list
static bool listed(const char *list, const char *name) {
    if (!list) return false;
    std::string l = std::string(",") + list + ",";
    return l.find(std::string(",") + name + ",") != std::string::npos;
}
static std::string g_only, g_only_variant;
static bool g_listed_only = false, g_all_variants = false;   // log suite: run a scenario only for the must-pass variants
                                                              // and its own must-fail ones (--all: every combination)
static std::string drops(Env &e, int h) {
    Stats &x = e.stats(h);
    return "host" + std::to_string(h) + " fill drops: tok=" + std::to_string(x.fill_drop_tok) + " pending=" +
           std::to_string(x.fill_drop_pending) + " cached=" + std::to_string(x.fill_drop_cached) + " space=" +
           std::to_string(x.fill_drop_space) + " restart=" + std::to_string(x.fill_drop_restart) + " admits=" +
           std::to_string(x.admits);
}
static int run_suite(const char *suite, const char *must_pass /* comma list */, const std::vector<Variant> &vars,
                     const std::vector<Scenario> &scen, bool verbose) {
    int bad = 0;
    printf("## suite: %s\n", suite);
    printf("%-14s %-21s %-8s %-11s %s\n", "scenario", "variant", "errors", "interleave", "verdict");
    for (auto &s : scen)
        for (auto &v : vars) {
            if (!g_only.empty() && g_only != s.name) continue;
            if (!g_only_variant.empty() && g_only_variant != v.name) continue;
            bool want_fail = listed(s.must_fail_in, v.name);
            bool must_ok = listed(must_pass, v.name);
            if (g_listed_only && !g_all_variants && !want_fail && !must_ok) continue;
            g_committed = committed_rules(v.r);
            Result res = s.fn(v.r);
            g_committed = false;
            const char *verdict = "info";
            if (must_ok && !want_fail) verdict = (res.errors == 0 && res.reproduced) ? "PASS" : "FAIL";
            if (want_fail) verdict = (res.errors > 0 && res.reproduced) ? "PASS (violation detected)" : "FAIL (no violation)";
            if (!strncmp(verdict, "FAIL", 4)) bad++;
            printf("%-14s %-21s %-8lu %-11s %s\n", s.name, v.name, (unsigned long)res.errors,
                   res.reproduced ? "forced" : "NOT forced", verdict);
            if (verbose || !strncmp(verdict, "FAIL", 4) || (want_fail && s.fn != basic)) {
                for (auto &l : res.log) printf("      %s\n", l.c_str());
                for (auto &l : res.errs) printf("      checker: %s\n", l.c_str());
            }
            fflush(stdout);
        }
    return bad;
}

int main(int argc, char **argv) {
    bool verbose = false;
    std::string only;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-v") verbose = true;
        else if (a == "--procs") g_procs = true;
        else if (a == "--suite" && i + 1 < argc) only = argv[++i];
        else if (a == "--only" && i + 1 < argc) g_only = argv[++i];
        else if (a == "--variant" && i + 1 < argc) g_only_variant = argv[++i];
        else if (a == "--all") g_all_variants = true;
        else if (a == "--emu") g_force_emu = true;
        else if (a == "--li16") {}   // already the default
        else if (a == "--li32") g_force_li32 = true;
        else if (a == "--put-key-after" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v != "publish" && v != "reply") { fprintf(stderr, "--put-key-after reply|publish\n"); return 2; }
            g_put_key_after = v == "publish";
        }
        else if (a == "--put-async" && i + 1 < argc) g_put_async = atoi(argv[++i]);
        else if (a == "--obj-read" && i + 1 < argc) {
            if ((g_obj_read = obj_read_mode_parse(argv[++i])) < 0) { fprintf(stderr, "--obj-read copy|prefetch|mstore|both\n"); return 2; }
        }
        else if (a == "--hit-clock" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "rdtscp") g_hit_clock = 1; else if (v == "lfence3") g_hit_clock = 0; else { fprintf(stderr, "--hit-clock lfence3|rdtscp\n"); return 2; }
        }
        else if (a == "--li16-fp-bits" && i + 1 < argc) g_li16_fp_bits = atoi(argv[++i]);
        else if (a == "--li16-ep-bits" && i + 1 < argc) g_li16_ep_bits = atoi(argv[++i]);
        else if (a == "--ro-peers" && i + 1 < argc) g_ro_peers = atoi(argv[++i]) != 0;
        else if (a == "--xtra-dir-replicas" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v != "0" && v != "1" && v != "2") { fprintf(stderr, "--xtra-dir-replicas 0|1|2\n"); return 2; }
            g_xrep = v[0] - '0';
        }
        else if (a == "--mono" && i + 1 < argc) {
            g_mono = atoi(argv[++i]) != 0;
            if (g_mono && !RC_MONO) { fprintf(stderr, "--mono 1: this build has RC_MONO=0 (RC_MONO=1 bash proto/build.sh)\n"); return 2; }
        }
        else { fprintf(stderr, "usage: tests [-v] [--procs] [--suite revoke|lease|log|feed|xtra|delete|ro|mono] [--xtra-dir-replicas 0|1|2] [--ro-peers 0|1 (other hosts' blocks mapped read-only, with --procs)] [--mono 0|1 (monotonic reads wherever allowed)] [--only SCENARIO] [--variant V] [--all] [--emu] [--li32 (32 B local index; 16 B is the default)] [--hit-clock lfence3|rdtscp] [--obj-read copy|prefetch|mstore|both] [--li16-fp-bits 20] [--li16-ep-bits 16] [--put-async 0|1] [--put-key-after reply|publish]\n"); return 2; }
    }
    printf("# hit check clock: %s; obj_read: %s\n", g_hit_clock ? "rdtscp" : "lfence3",
           obj_read_mode_name(g_obj_read >= 0 ? g_obj_read : Config().obj_read_mode));
    printf("# mode: %s%s%s\n", g_procs ? "procs (one process per host, POSIX shm)" : "threads (one process)",
           g_force_emu ? ", software non-coherent reader caches in every scenario" : "",
           !g_force_li32 ? (std::string(", 16 B local-index entries in log scenarios (default), fingerprint bits ") + std::to_string(g_li16_fp_bits) +
                            ", ep bits " + std::to_string(g_li16_ep_bits)).c_str() : "");
    if (g_ro_peers && !g_procs) { fprintf(stderr, "--ro-peers 1 needs --procs (one process per host)\n"); return 2; }
    if (g_ro_peers) printf("# ro_peers: every host process maps the other host's block read-only (ro_peers)\n");
    if (g_xrep) printf("# XREP: a directory copy per host wherever XTRA-style own freshness runs (xtra_dir_replicas=%d)\n", g_xrep);
    set_hook(gate_hook);
    int bad = 0;
    // ---- suite 1: the revocation protocol: lease off, revoke on (--update invalidate)
    if (only.empty() || only == "revoke") {
        Rules fixed;
        set_update(fixed, "invalidate");
        Rules notomb = fixed;   notomb.tomb = false;
        Rules nobump2 = fixed;  nobump2.bump2 = false;
        Rules nopend = fixed;   nopend.pending = false;
        Rules instfirst = fixed; instfirst.flush_before_install = false;
        Rules nodead = fixed;   nodead.slow_deadline = false;
        Rules nounpub = fixed;  nounpub.unpublish_on_put = false;
        Rules privtok = fixed;  privtok.priv_writer_token = false;
        std::vector<Variant> vars = {{"fixed", fixed}, {"fixed-notomb", notomb}, {"fixed-nobump2", nobump2},
                                     {"fixed-nopending", nopend}, {"fixed-installfirst", instfirst},
                                     {"fixed-nodeadline", nodead}, {"fixed-nounpublish", nounpub},
                                     {"fixed-privinstalltok", privtok}};
        std::vector<Scenario> scen = {
            {"basic", basic, nullptr},
            {"tombstone", scenario_tomb, "fixed-notomb"},
            {"bump2", scenario_bump2, "fixed-nobump2"},
            {"pending", scenario_pending, "fixed-nopending"},
            {"install", scenario_install, "fixed-installfirst"},
            {"deadline", scenario_deadline, "fixed-nodeadline"},
            {"slowput", scenario_slowput, "fixed-nounpublish"},
            {"privwriter", scenario_privwriter, "fixed-privinstalltok"},
        };
        bad += run_suite("revoke (lease=0 revoke=1)", "fixed", vars, scen, verbose);
    }
    // ---- suite 2: leases (--update lease)
    if (only.empty() || only == "lease") {
        g_tweak = [](Config &c) { c.lease_ns = TL; };
        Rules L;
        set_update(L, "lease");
        Rules atinst = L;  atinst.lease_from_ai_read = false;
        Rules nowait = L;  nowait.put_wait = false;
        Rules nobump2 = L; nobump2.bump2 = false;
        Rules notok = L;   notok.filltok = false;
        Rules nopend = L;  nopend.pending = false;
        Rules nounpub = L; nounpub.unpublish_on_put = false;
        Rules notsc = L;   notsc.lease_tsc = false;
        Rules nobr = L;    nobr.lease_bracket = false;
        Rules norst = L;   norst.restart_wait = false;
        Rules fromend = L; fromend.put_wait_from_cut = false;
        Rules instfirst = L; instfirst.flush_before_install = false;
        Rules nolease = L; nolease.lease = false;
        Rules norestamp = L; norestamp.restamp = false;
        std::vector<Variant> vars = {{"lease", L}, {"lease-at-install", atinst}, {"lease-nowait", nowait},
                                     {"lease-nobump2", nobump2}, {"lease-nofilltok", notok}, {"lease-nopending", nopend},
                                     {"lease-nounpublish", nounpub}, {"lease-notsc", notsc}, {"lease-nobracket", nobr},
                                     {"lease-norestartwait", norst}, {"lease-waitfromend", fromend},
                                     {"lease-installfirst", instfirst}, {"lease-norestamp", norestamp}, {"no-lease", nolease}};
        std::vector<Scenario> scen = {
            {"basic", basic, nullptr},
            {"L1-atinstall", lease_atinstall, "lease-at-install"},
            {"L2-putreturn", lease_putreturn, "lease-nowait,no-lease"},
            {"L3-stalefill", lease_stalefill, "lease-nobump2,lease-nofilltok"},
            {"L4-fillinput", lease_fillduringput, "lease-nopending"},
            {"L5-hotrewrite", lease_hotrewrite, "no-lease"},
            {"L6-epochstall", lease_epochstall, "lease-notsc"},
            {"L7-stall-limit", lease_stall_limit, "lease"},
            {"L8-unreachable", lease_unreachable, nullptr},
            {"L9-restart", lease_restart, nullptr},
            {"L10-bracket", lease_bracket, nullptr},
            // the revocation-suite scenarios must also hold under leases
            {"tombstone", scenario_tomb, nullptr},
            // (bump2 is left out: under leases the reader hits instead of missing, so its interleaving cannot be
            // forced; L3 is its lease counterpart)
            {"pending", scenario_pending, nullptr},
            {"install", scenario_install, "lease-installfirst"},
            {"deadline", scenario_deadline, nullptr},
            {"slowput", scenario_slowput, nullptr},
            {"privwriter", scenario_privwriter, nullptr},
            {"W3-restamp", wrap_restamp, "lease-norestamp"},
        };
        bad += run_suite("lease (lease=1 revoke=0, T=400 ms)", "lease", vars, scen, verbose);
        g_tweak = nullptr;
    }
    // ---- suite 3: owner log + host polling without the change feed (--update log, Rules::feed off)
    if (only.empty() || only == "log") {
        // allocations are driven step by step (tiny gen_bits in W*): no sweep-margin check
        g_tweak = [](Config &c) { c.log_poll_ns = TP; c.li_slots = 4096; c.log_sweep_slots = 0; c.log_max_alloc_per_s = 0; };
        g_listed_only = true;
        Rules G;
        set_update(G, "log");
        G.feed = false;     // the lazy log, fill tokens by message (the change feed has its own suite below)
        G.kg_cxl = false;
        auto var = [&](void (*f)(Rules &)) { Rules x = G; f(x); return x; };
        std::vector<Variant> vars = {
            {"log", G},
            {"log-seqlock", var([](Rules &x) { x.log_seqlock = true; })},
            {"log-hbafter", var([](Rules &x) { x.log_hb_before = false; })},
            {"log-tsafter", var([](Rules &x) { x.log_ts_before_tails = false; })},
            {"log-waitfromstart", var([](Rules &x) { x.log_wait_from_cut = false; })},
            {"log-concurrentput", var([](Rules &x) { x.put_serial = false; })},
            {"log-noinstallcheck", var([](Rules &x) { x.log_install_check = false; })},
            {"log-snapafter", var([](Rules &x) { x.log_snap_before = false; })},
            {"log-appendfirst", var([](Rules &x) { x.log_unpublish_first = false; })},
            {"log-atpublish", var([](Rules &x) { x.log_every_put = false; })},
            {"log-nowait", var([](Rules &x) { x.put_wait = false; })},
            // the swap order (Rules::put_cut_at_publish): log-swap = RAW by waiting W, log-swap-nowait = Krater's committed
            // ack; both must pass (bounded staleness checked)
            {"log-swap", var([](Rules &x) { x.put_cut_at_publish = true; })},
            {"log-swap-nowait", var([](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; })},
            // controls of the swap order:
            {"log-swap-nofilltok", var([](Rules &x) { x.put_cut_at_publish = true; x.filltok = false; })},
            {"log-swap-nowait-noclock", var([](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; x.log_hit_clock = false; })},
            {"log-swap-delfirst", var([](Rules &x) { x.put_cut_at_publish = true; x.swap_del_first = true; })},
            {"log-swap-nowait-delfirst", var([](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; x.swap_del_first = true; })},
            {"log-swap-nopending", var([](Rules &x) { x.put_cut_at_publish = true; x.pending = false; })},   // info (redundant)
            {"log-noclock", var([](Rules &x) { x.log_hit_clock = false; })},
            {"log-nosweep", var([](Rules &x) { x.log_sweep = false; })},
            {"log-nodropall", var([](Rules &x) { x.log_dropall_lag = false; })},
            {"log-norestamp", var([](Rules &x) { x.restamp = false; })},
            {"log-nooverrun", var([](Rules &x) { x.log_overrun = false; })},
            {"log-noepoch", var([](Rules &x) { x.log_epoch = false; })},
            {"log-nobump2", var([](Rules &x) { x.bump2 = false; })},
            {"log-nofilltok", var([](Rules &x) { x.filltok = false; })},
            {"log-nopending", var([](Rules &x) { x.pending = false; })},
            {"log-installfirst", var([](Rules &x) { x.flush_before_install = false; })},
            {"no-log", var([](Rules &x) { x.log = false; })},
        };
        std::vector<Scenario> scen = {
            {"basic", basic, nullptr},
            {"G1-putreturn", log_putreturn, "log-nowait,no-log"},
            {"G2-hbafter", log_hbafter, "log-hbafter"},
            {"M4-tsbefore", log_tsbefore, "log-tsafter"},
            {"M4-waitfromcut", log_waitfromcut, "log-waitfromstart"},
            {"M5-serial", put_serial, "log-concurrentput"},
            {"A1-asyncput", async_putreturn, nullptr},
            {"G3-installchk", log_installcheck, "log-noinstallcheck,log-snapafter"},
            {"G4-appendfirst", log_appendfirst, "log-appendfirst"},
            {"G5-atpublish", log_atpublish, "log-atpublish"},
            {"G6-pollstall", log_pollstall, "log-noclock"},
            {"G7-overrun", log_overrun, "log-nooverrun"},
            {"G8-epoch", log_epoch, "log-noepoch"},
            {"W1-sweep", wrap_sweep, "log-nosweep"},
            {"W2-dropall", wrap_dropall, "log-nodropall"},
            {"W3-restamp", wrap_restamp, "log-norestamp"},
            {"W4-midstall", wrap_midstall, "log-noclock"},
            // the fill rules are unchanged under the log, and the revocation / lease scenarios must hold too
            {"L3-stalefill", lease_stalefill, "log-nobump2,log-nofilltok,log-swap-nofilltok"},
            {"L4-fillinput", lease_fillduringput, "log-nopending"},
            {"C1-boundedstale", committed_boundedstale, "log-swap-nowait-noclock"},
            {"C2-swaporder", swap_order, "log-swap-delfirst,log-swap-nowait-delfirst"},
            {"tombstone", scenario_tomb, nullptr},
            {"pending", scenario_pending, nullptr},
            {"install", scenario_install, "log-installfirst"},
            {"deadline", scenario_deadline, nullptr},
            {"slowput", scenario_slowput, nullptr},
            {"privwriter", scenario_privwriter, nullptr},
            {"L8-unreachable", lease_unreachable, nullptr},
            {"L9-restart", lease_restart, nullptr},
            {"U1-unreachget", unreachable_get, nullptr},
        };
        bad += run_suite("log (log=1, P=W=200 ms)", "log,log-seqlock,log-swap,log-swap-nowait", vars, scen, verbose);
        g_tweak = nullptr;
        g_listed_only = false;
    }
    // ---- suite 4: change feed + kg read from CXL + local-index cap (the default protocol). Every log scenario must still
    //      hold; F1-F4 are the counterexamples of the feed's own rules.
    if (only.empty() || only == "feed") {
        // allocations are driven step by step (tiny gen_bits in W*): no sweep-margin check
        g_tweak = [](Config &c) { c.log_poll_ns = TP; c.li_slots = 4096; c.log_sweep_slots = 0; c.log_max_alloc_per_s = 0; };
        g_listed_only = true;
        Rules F;   // defaults: log + feed + kg_cxl
        auto var = [&](void (*f)(Rules &)) { Rules x = F; f(x); return x; };
        std::vector<Variant> vars = {
            {"feed", F},
            {"feed-nodel", var([](Rules &x) { x.feed_del = false; })},
            {"feed-noinstallcheck", var([](Rules &x) { x.log_install_check = false; })},
            {"kg-late", var([](Rules &x) { x.kg_early = false; })},
            {"refresh-noflush", var([](Rules &x) { x.refresh_flush = false; })},
            {"feed-nowait", var([](Rules &x) { x.put_wait = false; })},
            {"feed-swap", var([](Rules &x) { x.put_cut_at_publish = true; })},   // the swap order (see the log suite)
            {"feed-swap-nowait", var([](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; })},
            // the swap's other record order (refresh, then delete: swap_refresh_after_del = false) must pass too
            {"feed-swap-bird", var([](Rules &x) { x.put_cut_at_publish = true; x.swap_refresh_after_del = false; })},
            {"feed-swap-nowait-bird", var([](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; x.swap_refresh_after_del = false; })},
            {"feed-swap-nofilltok", var([](Rules &x) { x.put_cut_at_publish = true; x.filltok = false; })},
            {"feed-swap-nowait-noclock", var([](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; x.log_hit_clock = false; })},
            {"feed-swap-delfirst", var([](Rules &x) { x.put_cut_at_publish = true; x.swap_del_first = true; })},
            {"feed-swap-nowait-delfirst", var([](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; x.swap_del_first = true; })},
            {"feed-swap-nopending", var([](Rules &x) { x.put_cut_at_publish = true; x.pending = false; })},   // info (redundant)
            {"feed-tsafter", var([](Rules &x) { x.log_ts_before_tails = false; })},
            {"feed-waitfromstart", var([](Rules &x) { x.log_wait_from_cut = false; })},
            {"feed-concurrentput", var([](Rules &x) { x.put_serial = false; })},
            {"feed-nosweep", var([](Rules &x) { x.log_sweep = false; })},
            {"feed-nodropall", var([](Rules &x) { x.log_dropall_lag = false; })},
            {"feed-norestamp", var([](Rules &x) { x.restamp = false; })},
            {"feed-noclock", var([](Rules &x) { x.log_hit_clock = false; })},
            {"feed-nobump2", var([](Rules &x) { x.bump2 = false; })},
            {"feed-nopending", var([](Rules &x) { x.pending = false; })},
            {"feed-erasefailed", var([](Rules &x) { x.keep_failed = false; })},
        };
        std::vector<Scenario> scen = {
            {"basic", basic, nullptr},
            {"F1-nodel", log_atpublish, "feed-nodel"},
            {"F2-installchk", log_installcheck, "feed-noinstallcheck"},
            {"F3-kglate", feed_kglate, "kg-late"},
            {"F4-refreshflush", feed_refreshflush, "refresh-noflush"},
            {"F5-keepfailed", feed_keepfailed, nullptr},   // fast hit expected with keep_failed (the default)
            {"G1-putreturn", log_putreturn, "feed-nowait"},
            {"G2-hbafter", log_hbafter, nullptr},
            {"M4-tsbefore", log_tsbefore, "feed-tsafter"},
            {"M4-waitfromcut", log_waitfromcut, "feed-waitfromstart"},
            {"M5-serial", put_serial, "feed-concurrentput"},
            {"A1-asyncput", async_putreturn, nullptr},
            {"G4-appendfirst", log_appendfirst, nullptr},
            {"G6-pollstall", log_pollstall, "feed-noclock"},
            {"G7-overrun", log_overrun, nullptr},
            {"G8-epoch", log_epoch, nullptr},
            {"W1-sweep", wrap_sweep, "feed-nosweep"},
            {"W2-dropall", wrap_dropall, "feed-nodropall"},
            {"W3-restamp", wrap_restamp, "feed-norestamp"},
            {"W4-midstall", wrap_midstall, "feed-noclock"},
            {"L3-stalefill", lease_stalefill, "feed-nobump2,feed-swap-nofilltok"},
            {"L4-fillinput", lease_fillduringput, "feed-nopending"},
            {"C1-boundedstale", committed_boundedstale, "feed-swap-nowait-noclock"},
            {"C2-swaporder", swap_order, "feed-swap-delfirst,feed-swap-nowait-delfirst"},
            {"install", scenario_install, nullptr},
            {"tombstone", scenario_tomb, nullptr},
            {"slowput", scenario_slowput, nullptr},
            {"L8-unreachable", lease_unreachable, nullptr},
            {"L9-restart", lease_restart, nullptr},
            {"U1-unreachget", unreachable_get, nullptr},
        };
        bad += run_suite("feed (log + change feed + kg in CXL + local-index cap, P=W=200 ms)", "feed,feed-erasefailed,feed-swap,feed-swap-nowait,feed-swap-bird,feed-swap-nowait-bird", vars, scen, verbose);
        g_tweak = nullptr;
        g_listed_only = false;
    }
    // ---- suite 5: XTRA-style with its own freshness check (Config::xtra_own_freshness): hits compare
    //      the directory word with the entry's generation; no log records, no heartbeat; swap order, RAW without W
    if (only.empty() || only == "xtra") {
        g_scheme = Scheme::XTRA;
        g_tweak = [](Config &c) { c.log_poll_ns = TP; c.li_slots = 4096; c.log_sweep_slots = 0; c.log_max_alloc_per_s = 0;
                                  c.xtra_own_freshness = true; };
        g_listed_only = true;
        Rules X;   // defaults (feed and kg_cxl are moot: no records); put_cut_at_publish names the swap order the PUTs take
        X.put_cut_at_publish = true;
        auto var = [&](void (*f)(Rules &)) { Rules x = X; f(x); return x; };
        std::vector<Variant> vars = {
            {"xtra-own", X},
            {"xtra-own-nokillput", var([](Rules &x) { x.xtra_kill_on_put = false; })},
            {"xtra-own-nokillevict", var([](Rules &x) { x.xtra_kill_on_evict = false; })},
            {"xtra-own-nofilltok", var([](Rules &x) { x.filltok = false; })},
        };
        std::vector<Scenario> scen = {
            {"basic", basic, nullptr},
            {"G1-putreturn", log_putreturn, "xtra-own-nokillput"},
            {"X2-evictput", xtra_evictput, nullptr},   // directory versions: the PUT moves the key's slot; nokillevict is info
            {"X3-collide", xtra_collide, nullptr},
            {"X4-clockq", xtra_clockq, nullptr},   // Config::li_clock_inq (the queue does not grow on revalidation)
            {"C1-boundedstale", committed_boundedstale, nullptr},
            {"C2-swaporder", swap_order, nullptr},
            {"L3-stalefill", lease_stalefill, "xtra-own-nofilltok"},
            {"L4-fillinput", lease_fillduringput, nullptr},
            {"install", scenario_install, nullptr},
            {"U1-unreachget", unreachable_get, nullptr},
        };
        bad += run_suite("xtra (XTRA-style, own freshness: directory word, no log / poller / heartbeat)", "xtra-own", vars, scen, verbose);
        g_tweak = nullptr;
        g_scheme = Scheme::RACKCACHE;
        g_listed_only = false;
    }
    // ---- suite 6: DELETE in every scheme (Client::del / expire; controls: the del-nounpub / del-nobump / del-norec variants
    //      and swap_del_first). Krater, XTRA-style (own freshness and borrowed), the Beluga / MEGALON-style baselines
    //      (Krater's path), PRIV (invalidate every host), SHARD (the owner deletes), and the lease / invalidate mechanisms.
    if (only.empty() || only == "delete") {
        const std::function<void(Config &)> logtw = [](Config &c) { c.log_poll_ns = TP; c.li_slots = 4096; c.log_sweep_slots = 0;
                                                                     c.log_max_alloc_per_s = 0; };
        g_listed_only = true;
        Rules F;   // defaults: log + feed + kg in CXL, window order (RAW by the window)
        auto var = [&](const Rules &b, void (*f)(Rules &)) { Rules x = b; f(x); return x; };
        Rules SW = var(F, [](Rules &x) { x.put_cut_at_publish = true; });                       // swap + W (RAW by waiting)
        Rules SN = var(F, [](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; });   // swap, committed
        Rules V1 = var(F, [](Rules &x) { x.feed = false; x.kg_cxl = false; });                  // lazy log, tokens by message
        auto norec = [](Rules &x) { x.del_record = false; };
        auto nounpub = [](Rules &x) { x.del_unpublish = false; };
        auto nobump = [](Rules &x) { x.del_bump = false; };
        auto scen = [](const char *d1, const char *d2, const char *d5, const char *d6) {
            return std::vector<Scenario>{{"D1-delete", del_raw, d1},     {"D2-stalefill", del_stalefill, d2},
                                         {"D3-expire", del_expire, nullptr}, {"D4-delput", del_delput, nullptr},
                                         {"D5-swapmid", del_swapmid, d5}, {"D6-boundedstale", del_boundedstale, d6}};
        };
        g_scheme = Scheme::RACKCACHE;
        g_tweak = logtw;
        {
            std::vector<Variant> vars = {
                {"feed", F}, {"feed-swap", SW}, {"feed-swap-nowait", SN},
                {"log", V1}, {"log-swap-nowait", var(V1, [](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; })},
                {"feed-del-nounpub", var(F, nounpub)}, {"feed-del-nobump", var(F, nobump)}, {"feed-del-norec", var(F, norec)},
                {"feed-swap-del-nounpub", var(SW, nounpub)}, {"feed-swap-del-nobump", var(SW, nobump)},
                {"feed-swap-del-norec", var(SW, norec)},
                {"feed-swap-nowait-del-nounpub", var(SN, nounpub)}, {"feed-swap-nowait-del-nobump", var(SN, nobump)},
                {"feed-swap-nowait-del-norec", var(SN, norec)},
                {"feed-swap-delfirst", var(SW, [](Rules &x) { x.swap_del_first = true; })},
                {"feed-swap-nowait-delfirst", var(SN, [](Rules &x) { x.swap_del_first = true; })},
                {"feed-swap-nowait-noclock", var(SN, [](Rules &x) { x.log_hit_clock = false; })},
            };
            bad += run_suite("delete / Krater (P=W=200 ms)", "feed,feed-swap,feed-swap-nowait,log,log-swap-nowait", vars,
                             scen("feed-del-nounpub,feed-del-norec,feed-swap-del-nounpub,feed-swap-del-norec,"
                                  "feed-swap-nowait-del-nounpub,feed-swap-nowait-del-norec",
                                  "feed-del-nobump,feed-swap-del-nobump,feed-swap-nowait-del-nobump",
                                  "feed-swap-delfirst,feed-swap-nowait-delfirst", "feed-swap-nowait-noclock"),
                             verbose);
        }
        g_scheme = Scheme::XTRA;
        g_tweak = [logtw](Config &c) { logtw(c); c.xtra_own_freshness = true; };
        {
            Rules X = SW;   // own freshness: the swap order, RAW without W
            std::vector<Variant> vars = {{"xtra-own", X}, {"xtra-own-nokillput", var(X, [](Rules &x) { x.xtra_kill_on_put = false; })},
                                         {"xtra-own-del-nounpub", var(X, nounpub)}, {"xtra-own-del-nobump", var(X, nobump)}};
            std::vector<Scenario> sc = scen("xtra-own-nokillput,xtra-own-del-nounpub", "xtra-own-del-nobump", nullptr, nullptr);
            sc.erase(sc.begin() + 4);   // D5: no records (nothing between the switch and the cut)
            bad += run_suite("delete / XTRA-style own freshness", "xtra-own", vars, sc, verbose);
        }
        // the baselines that take Krater's path (window order, and swap + committed), XTRA-style borrowed included
        g_tweak = logtw;
        for (Scheme sch : {Scheme::CXL_UC, Scheme::GSTATE_CXL, Scheme::GSTATE_COH, Scheme::VER, Scheme::XTRA}) {
            g_scheme = sch;
            std::string n = scheme_name(sch), w = n + "-swap-nowait", c = n + "-del-norec";
            static std::deque<std::string> names;   // the suite tables hold C strings
            auto keep = [](std::string x) { names.push_back(std::move(x)); return names.back().c_str(); };
            std::vector<Variant> vars = {{keep(n), F}, {keep(w), SN}, {keep(c), var(F, norec)}};
            // VER has no local index: the record is not needed (its norec run is info)
            const bool li = sch != Scheme::VER;
            bad += run_suite(keep("delete / " + n + " (Krater's path)"), keep(n + "," + w), vars,
                             scen(li ? keep(c) : nullptr, nullptr, nullptr, nullptr), verbose);
        }
        g_tweak = nullptr;
        g_scheme = Scheme::PRIV;   // every host caches privately: a DELETE invalidates every host (two rounds)
        {
            std::vector<Variant> vars = {{"priv", F}, {"priv-del-norec", var(F, norec)}};
            std::vector<Scenario> sc = scen("priv-del-norec", nullptr, nullptr, nullptr);
            sc.erase(sc.begin() + 4, sc.end());   // D5 / D6: the log scheme's
            bad += run_suite("delete / PRIV", "priv", vars, sc, verbose);
        }
        g_scheme = Scheme::SHARD;   // the owner holds the only copy and deletes it
        {
            std::vector<Variant> vars = {{"shard", F}, {"shard-swap", var(F, [](Rules &x) { x.put_cut_at_publish = true; })}};
            std::vector<Scenario> sc = scen(nullptr, nullptr, nullptr, nullptr);
            sc.erase(sc.begin() + 4, sc.end());
            bad += run_suite("delete / SHARD", "shard,shard-swap", vars, sc, verbose);
        }
        g_scheme = Scheme::RACKCACHE;
        {
            Rules I;
            set_update(I, "invalidate");
            std::vector<Variant> vars = {{"invalidate", I}, {"invalidate-del-norec", var(I, norec)}};
            std::vector<Scenario> sc = scen("invalidate-del-norec", nullptr, nullptr, nullptr);
            sc.erase(sc.begin() + 4, sc.end());
            bad += run_suite("delete / invalidate (lease=0 revoke=1)", "invalidate", vars, sc, verbose);
        }
        g_tweak = [](Config &c) { c.lease_ns = TL; };
        {
            Rules L;
            set_update(L, "lease");
            std::vector<Variant> vars = {{"lease", L}, {"lease-del-nounpub", var(L, nounpub)}};
            std::vector<Scenario> sc = scen("lease-del-nounpub", nullptr, nullptr, nullptr);
            sc.erase(sc.begin() + 4, sc.end());
            bad += run_suite("delete / lease (T=400 ms)", "lease", vars, sc, verbose);
        }
        g_tweak = nullptr;
        g_listed_only = false;
    }
    // ---- suite 7: other owners' blocks mapped read-only (Config::ro_peers); forked children, any mode
    if (only.empty() || only == "ro") {
        Rules F;
        std::vector<Scenario> sc = {{"R1-peerwrite", ro_peerwrite, "rw-peers"}, {"R2-ownwrite", ro_ownwrite, nullptr}};
        g_tweak = [](Config &c) { c.ro_peers = true; };
        bad += run_suite("ro (the other host's block read-only)", "ro-peers", {{"ro-peers", F}}, sc, verbose);
        g_tweak = [](Config &c) { c.ro_peers = false; };   // control: ro_peers off, the default (read-write everywhere)
        bad += run_suite("ro control (read-write mapping)", "rw-peers", {{"rw-peers", F}}, sc, verbose);
        g_tweak = nullptr;
    }
    // ---- suite 8: monotonic reads (P6) per host, Config::mono; P6 counts as an error here
    //      (only in an RC_MONO=1 build: RC_MONO=1 bash proto/build.sh -> build-$(hostname)-mono1/tests)
    if (!RC_MONO && only == "mono") printf("## suite: mono -- skipped: this build has RC_MONO=0 (Config::mono compiled out)\n");
    if (RC_MONO && (only.empty() || only == "mono")) {
        const std::function<void(Config &)> logtw = [](Config &c) { c.log_poll_ns = TP; c.li_slots = 4096; c.log_sweep_slots = 0;
                                                                     c.log_max_alloc_per_s = 0; };
        g_listed_only = true;
        g_check_mono = true;
        Rules F;
        auto var = [&](const Rules &b, void (*f)(Rules &)) { Rules x = b; f(x); return x; };
        Rules SN = var(F, [](Rules &x) { x.put_cut_at_publish = true; x.put_wait = false; });   // swap, committed
        Rules SW = var(F, [](Rules &x) { x.put_cut_at_publish = true; });                       // swap + W
        auto early = [](Rules &x) { x.mono_hold = false; };
        g_tweak = logtw;   // Config::mono off: M1-ghost must violate P6 (no write path gives monotonic reads by itself)
        bad += run_suite("mono off (P6 checked)", "", {{"feed-swap-nowait", SN}, {"feed-swap", SW}},
                         {{"M1-ghost", mono_ghost, "feed-swap-nowait,feed-swap"}}, verbose);
        g_tweak = [logtw](Config &c) { logtw(c); c.mono = true; };
        {
            std::vector<Variant> vars = {{"feed-swap-nowait-mono", SN}, {"feed-swap-mono", SW}, {"feed-mono", F},
                                         {"feed-swap-nowait-mono-early", var(SN, early)}, {"feed-swap-mono-early", var(SW, early)}};
            std::vector<Scenario> sc = {{"M1-ghost", mono_ghost, "feed-swap-nowait-mono-early,feed-swap-mono-early"},
                                        {"basic", basic, nullptr}, {"G1-putreturn", log_putreturn, nullptr},
                                        {"C1-boundedstale", committed_boundedstale, nullptr}, {"L3-stalefill", lease_stalefill, nullptr},
                                        {"D1-delete", del_raw, nullptr}, {"D2-stalefill", del_stalefill, nullptr}};
            bad += run_suite("mono on (P6 checked)", "feed-swap-nowait-mono,feed-swap-mono,feed-mono", vars, sc, verbose);
        }
        g_tweak = nullptr;
        g_check_mono = false;
        g_listed_only = false;
    }
    set_hook(HookFn());
    printf("%s\n", bad ? "SOME EXPECTATIONS FAILED" : "ALL EXPECTATIONS MET");
    return bad != 0;
}
