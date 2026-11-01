// Simulated backend and the independent GET checker.
// Value of (key, version) is deterministic: [key u64][ver u32][vlen u32][pseudo-random bytes].
// Checker, for every GET: P1 the value belongs to the requested key; bytes match (key, ver) exactly;
// P2 ver >= the version of the last PUT(key) that completed before the GET started; ver <= backend's.
// P3 (owner side, via Config::on_publish): an AI entry is never published with a version older than a
// PUT that has already completed. Optional: bounded staleness from the cut, via Config::on_cut (set_stale_bound()).
// DELETE: a delete is a version too (a tombstone: bit 31 of the backend's version word); put_done / on_cut
// take it as for a PUT, so a GET that returns an older value after the delete completed is a P2 error (bounded staleness:
// err_stale). An absent result (vlen 0, the header [key][ver][0]) is checked for P1 (key), P2 (ver >= the last completed
// write), not from the future, and: if the backend still holds version ver, it must be a tombstone.
#pragma once
#include <immintrin.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rackcache.h"

namespace rc {

inline uint64_t splitmix(uint64_t &s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
inline uint64_t now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
inline void spin_ns(uint64_t ns) {
    if (!ns) return;
    uint64_t end = now_ns() + ns;
    while (now_ns() < end) _mm_pause();
}

// 70% 16-199 B, 25% 200-999 B, 5% 1000-3999 B
inline uint32_t value_len(uint64_t key, uint32_t ver) {
    uint64_t s = key * 0x2545F4914F6CDD1DULL + ver;
    uint64_t h = splitmix(s);
    uint32_t r = h % 100, x = (uint32_t)(h >> 20);
    if (r < 70) return 16 + x % 184;
    if (r < 95) return 200 + x % 800;
    return 1000 + x % 3000;
}
// vl: value length (>= 16); default: value_len(key, ver)
inline uint32_t make_value(uint64_t key, uint32_t ver, char *out, uint32_t vl = 0) {
    if (!vl) vl = value_len(key, ver);
    memcpy(out, &key, 8);
    memcpy(out + 8, &ver, 4);
    memcpy(out + 12, &vl, 4);
    uint64_t s = key ^ ((uint64_t)ver << 40) ^ 0xA5A5A5A5ULL;
    for (uint32_t i = 16; i < vl; i += 8) {
        uint64_t r = splitmix(s);
        memcpy(out + i, &r, std::min<uint32_t>(8, vl - i));
    }
    return vl;
}

// Versions live in memory the backend owns (threads mode) or in a shared segment (one process per host: ext != nullptr,
// nkeys+1 entries). A version word is the version (bits 0-30) | DEL_BIT if that version is a DELETE's tombstone.
class SimBackend : public Backend {
public:
    SimBackend(uint64_t nkeys, uint64_t delay_ns, std::atomic<uint32_t> *ext = nullptr) : delay_(delay_ns) {
        if (!ext) {
            own_.reset(new std::atomic<uint32_t>[nkeys + 1]);
            for (uint64_t i = 0; i <= nkeys; i++) own_[i].store(0);
            ext = own_.get();
        }
        ver_ = ext;
    }
    static constexpr uint32_t DEL_BIT = 1u << 31, VER_MASK = DEL_BIT - 1;
    uint32_t get(uint64_t key, char *out, uint32_t *vlen) override {
        spin_ns(delay_);
        gets++;
        uint32_t w = ver_[key].load(std::memory_order_acquire), v = w & VER_MASK;
        if (w & DEL_BIT) {   // deleted: the header only, vlen 0
            const uint32_t z = 0;
            memcpy(out, &key, 8);
            memcpy(out + 8, &v, 4);
            memcpy(out + 12, &z, 4);
            *vlen = 0;
            return v;
        }
        *vlen = make_value(key, v, out, vlen_of(key, v));
        return v;
    }
    uint32_t put(uint64_t key, char *out, uint32_t *vlen) override {
        spin_ns(put_delay_ == ~0ULL ? delay_ : put_delay_);
        return put_complete(key, out, vlen);
    }
    uint64_t put_latency_ns(uint64_t key) override { return put_delay_ == ~0ULL ? delay_ : put_delay_; }
    uint32_t put_complete(uint64_t key, char *out, uint32_t *vlen) override {
        puts++;
        uint32_t v = next_version(key, 0);
        *vlen = make_value(key, v, out, vlen_of(key, v));
        return v;
    }
    uint32_t del(uint64_t key) override {
        spin_ns(put_delay_ == ~0ULL ? delay_ : put_delay_);
        return del_complete(key);
    }
    uint32_t del_complete(uint64_t key) override {
        dels++;
        return next_version(key, DEL_BIT);
    }
    uint32_t current(uint64_t key) const { return ver_[key].load(std::memory_order_acquire) & VER_MASK; }
    // the backend's current version of key is ver and it is a tombstone (true) / a value (false); other versions: -1
    int deleted_at(uint64_t key, uint32_t ver) const {
        uint32_t w = ver_[key].load(std::memory_order_acquire);
        return (w & VER_MASK) != ver ? -1 : (w & DEL_BIT) != 0;
    }
    void value_of(uint64_t key, uint32_t ver, char *out, uint32_t *vlen) override {
        *vlen = make_value(key, ver, out, vlen_of(key, ver));
    }
    // Trace replay: the value length of every version of key k is sizes[k] (at least 16 B, at most MAX_VLEN).
    void set_sizes(const uint32_t *sizes) { sizes_ = sizes; }
    uint32_t vlen_of(uint64_t key, uint32_t ver) const {
        if (!sizes_) return value_len(key, ver);
        return std::min<uint32_t>(MAX_VLEN, std::max<uint32_t>(16, sizes_[key]));
    }
    std::atomic<uint64_t> gets{0}, puts{0}, dels{0};

private:
    uint32_t next_version(uint64_t key, uint32_t del_bit) {   // one RMW per write (PUTs of a key may run concurrently)
        uint32_t w = ver_[key].load(std::memory_order_relaxed), n;
        do n = ((w & VER_MASK) + 1) | del_bit;
        while (!ver_[key].compare_exchange_weak(w, n, std::memory_order_acq_rel, std::memory_order_relaxed));
        return n & VER_MASK;
    }
    std::unique_ptr<std::atomic<uint32_t>[]> own_;
    std::atomic<uint32_t> *ver_;
    uint64_t delay_, put_delay_ = ~0ULL;
    const uint32_t *sizes_ = nullptr;
public:
    void set_put_delay(uint64_t ns) { put_delay_ = ns; }   // backend write latency, separate from the read latency
};

struct CheckCounters { std::atomic<uint64_t> gets{0}, puts{0}, err_p1{0}, err_p2{0}, err_p3{0}, err_bytes{0}, err_stale{0}, absent{0},
                                            err_mono{0}, mono_gets{0}; };   // P6 (monotonic reads): not in errors()

// done[k] (version of the last completed PUT) and the counters live in shared memory with one process per host.
class Checker {
public:
    // ext_cuts / ext_floor (one process per host): (nkeys + 1) * CUT_RING and (nkeys + 1) words for the bounded-staleness check
    Checker(uint64_t nkeys, SimBackend *be, std::atomic<uint32_t> *ext_done = nullptr, CheckCounters *ext_cnt = nullptr,
            std::atomic<uint64_t> *ext_cuts = nullptr, std::atomic<uint32_t> *ext_floor = nullptr)
        : be_(be), nkeys_(nkeys) {
        if (!ext_done) {
            own_done_.reset(new std::atomic<uint32_t>[nkeys + 1]);
            for (uint64_t i = 0; i <= nkeys; i++) own_done_[i].store(0);
            ext_done = own_done_.get();
        }
        done_ = ext_done;
        c_ = ext_cnt ? ext_cnt : &own_cnt_;
        if (!ext_cuts) {
            own_cuts_.reset(new std::atomic<uint64_t>[(nkeys + 1) * CUT_RING]);
            own_floor_.reset(new std::atomic<uint32_t>[nkeys + 1]);
            for (uint64_t i = 0; i < (nkeys + 1) * CUT_RING; i++) own_cuts_[i].store(0);
            for (uint64_t i = 0; i <= nkeys; i++) own_floor_[i].store(0);
            ext_cuts = own_cuts_.get();
            ext_floor = own_floor_.get();
        }
        cuts_ = ext_cuts;
        floor_ = ext_floor;
    }
    // BOUNDED STALENESS from the cut (G4; the committed ack's whole contract): Config::on_cut gives each
    // PUT's version and cut time; its deadline is max(cut + P, publish) + slack (clock-rate tolerance). A GET that STARTS
    // after a deadline must return that version or newer (err_stale). Judging by the start is implied by G4 (every check
    // of such a GET comes after cut + P) and never flags a legal read; GETs that straddle cut + P are not judged here
    // (the model checks them). Off unless set_stale_bound() is called (tests, log scheme). One 64-bit word per entry
    // (deadline in us << 20 | version), so a reader never sees a torn pair; a ring of CUT_RING per key (slot ver % ring).
    static constexpr int CUT_RING = 8;
    static constexpr uint64_t DL_MASK = (1ull << 44) - 1;   // the deadline field: us modulo 2^44
    void set_stale_bound(uint64_t p_ns, uint64_t slack_ns) { stale_p_ = p_ns; stale_slack_ = slack_ns; }
    void on_cut(uint64_t key, uint32_t ver, uint64_t cut_ns) {
        if (!stale_p_ || !ver || ver >= (1u << 20)) return;
        uint64_t dl = std::max(cut_ns + stale_p_, now_ns()) + stale_slack_;
        cuts_[key * CUT_RING + ver % CUT_RING].store((dl / 1000) << 20 | ver, std::memory_order_release);
    }
    uint32_t stale_begin(uint64_t key) {   // the version a GET that starts now must return at least
        if (!stale_p_) return 0;
        uint64_t now_us = now_ns() / 1000;
        tl_stale_start_us() = now_us;   // diagnosis: how late past the deadline a flagged GET started
        uint32_t f = floor_[key].load(std::memory_order_acquire);
        for (int i = 0; i < CUT_RING; i++) {
            uint64_t w = cuts_[key * CUT_RING + i].load(std::memory_order_acquire);
            // The deadline field keeps only the low 44 bits of the deadline in us (the << 20 drops the rest), and
            // CLOCK_MONOTONIC passes 2^44 us after 203.6 days of uptime: compared as plain numbers, every stored deadline
            // would then look already passed, and GETs within P after a cut would be flagged (false STALE errors). So
            // compare modulo 2^44: passed iff now - deadline is in [0, 2^43).
            if (w && ((now_us - (w >> 20)) & DL_MASK) < (DL_MASK >> 1) + 1) f = std::max<uint32_t>(f, (uint32_t)(w & 0xFFFFF));
        }
        uint32_t cur = floor_[key].load();
        while (cur < f && !floor_[key].compare_exchange_weak(cur, f)) {}
        return f;
    }
    static uint64_t &tl_stale_start_us() { static thread_local uint64_t v = 0; return v; }
    bool verify_stale(uint64_t key, uint32_t floor, uint32_t ver, int host, Kind kind) {
        if (ver >= floor) return true;
        // diagnosis: the GET's start and end relative to the deadline of the version it should have seen
        uint64_t w = cuts_[key * CUT_RING + floor % CUT_RING].load(std::memory_order_acquire), end_us = now_ns() / 1000;
        const bool have = (w & 0xFFFFF) == floor;
        auto rel = [&](uint64_t t) { int64_t d = (int64_t)(((t - (w >> 20)) & DL_MASK) << 20) >> 20; return (long long)d; };   // modulo 2^44, signed
        std::string late = !have ? " (deadline overwritten)" :
            " start-deadline=" + std::to_string(rel(tl_stale_start_us())) + "us end-deadline=" +
            std::to_string(rel(end_us)) + "us P=" + std::to_string(stale_p_ / 1000) + "us";
        return fail(c_->err_stale, key, host, kind, "STALE: returned v" + std::to_string(ver) + " but PUT v" +
                    std::to_string(floor) + " was cut more than P before the GET started" + late);
    }
    // P6: MONOTONIC READS per host -- a GET must not return an older version of a key than a GET of
    // the same host that COMPLETED before it started (concurrent GETs do not constrain each other). Per host and key, the
    // highest version a completed GET returned (host-local: each process keeps its own hosts'). Off unless enable_mono();
    // its errors (err_mono) are not in errors().
    void enable_mono(int hosts) {
        hwm_.clear();
        for (int h = 0; h < hosts; h++) {
            hwm_.emplace_back(new std::atomic<uint32_t>[nkeys_ + 1]);
            for (uint64_t k = 0; k <= nkeys_; k++) hwm_.back()[k].store(0, std::memory_order_relaxed);
        }
    }
    bool mono_on() const { return !hwm_.empty(); }
    uint32_t mono_begin(int host, uint64_t key) const { return hwm_.empty() ? 0 : hwm_[host][key].load(std::memory_order_acquire); }
    bool verify_mono(int host, uint64_t key, uint32_t floor, uint32_t ver, Kind kind) {
        if (hwm_.empty()) return true;
        c_->mono_gets++;
        std::atomic<uint32_t> &h = hwm_[host][key];
        uint32_t cur = h.load(std::memory_order_relaxed);
        while (cur < ver && !h.compare_exchange_weak(cur, ver, std::memory_order_acq_rel)) {}
        if (ver >= floor) return true;
        return fail(c_->err_mono, key, host, kind, "P6 (monotonic reads): returned v" + std::to_string(ver) + " after a GET of this host returned v" +
                    std::to_string(floor));
    }
    uint64_t mono_errors() const { return c_->err_mono; }
    const CheckCounters &counters() const { return *c_; }
    uint64_t p2_errors() const { return c_->err_p2; }
    uint64_t stale_errors() const { return c_->err_stale; }
    uint32_t begin(uint64_t key) const { return done_[key].load(std::memory_order_acquire); }
    void put_done(uint64_t key, uint32_t ver) {
        if (!ver) return;   // failed PUT (owner unreachable): nothing completed
        uint32_t cur = done_[key].load();
        while (cur < ver && !done_[key].compare_exchange_weak(cur, ver)) {}
        c_->puts++;
    }
    // returns true if the GET result is correct
    bool verify(uint64_t key, uint32_t req, const char *val, uint32_t vlen, int host, Kind kind) {
        c_->gets++;
        if (vlen == 0) return verify_absent(key, req, val, host, kind);
        if (vlen < 16) return fail(c_->err_bytes, key, host, kind, "short value vlen=" + std::to_string(vlen));
        uint64_t k;
        uint32_t v, vl;
        memcpy(&k, val, 8);
        memcpy(&v, val + 8, 4);
        memcpy(&vl, val + 12, 4);
        if (k != key) return fail(c_->err_p1, key, host, kind, "P1: value of key " + std::to_string(k));
        if (vl != vlen || vl != be_->vlen_of(k, v))
            return fail(c_->err_bytes, key, host, kind, "length mismatch vlen=" + std::to_string(vlen));
        char ref[MAX_OUT];
        make_value(k, v, ref, vl);
        if (memcmp(ref, val, vlen) != 0) return fail(c_->err_bytes, key, host, kind, "payload bytes differ, ver=" + std::to_string(v));
        if (v < req)
            return fail(c_->err_p2, key, host, kind,
                        "P2: returned v" + std::to_string(v) + " but PUT v" + std::to_string(req) + " completed before the GET");
        if (v > be_->current(key)) return fail(c_->err_bytes, key, host, kind, "version from the future v" + std::to_string(v));
        return true;
    }
    // an absent result (DELETE): the backend tombstone's header [key][ver][0]
    bool verify_absent(uint64_t key, uint32_t req, const char *val, int host, Kind kind) {
        c_->absent++;
        uint64_t k;
        uint32_t v, vl;
        memcpy(&k, val, 8);
        memcpy(&v, val + 8, 4);
        memcpy(&vl, val + 12, 4);
        if (k != key) return fail(c_->err_p1, key, host, kind, "P1: absent result of key " + std::to_string(k));
        if (vl != 0) return fail(c_->err_bytes, key, host, kind, "absent result with vlen field " + std::to_string(vl));
        if (kind != MISS) return fail(c_->err_bytes, key, host, kind, "absent result not from the backend");
        if (v < req)
            return fail(c_->err_p2, key, host, kind,
                        "P2: returned absent v" + std::to_string(v) + " but PUT v" + std::to_string(req) + " completed before the GET");
        if (v > be_->current(key)) return fail(c_->err_bytes, key, host, kind, "absent version from the future v" + std::to_string(v));
        if (be_->deleted_at(key, v) == 0)
            return fail(c_->err_bytes, key, host, kind, "absent v" + std::to_string(v) + " but the backend holds a value at v" + std::to_string(v));
        return true;
    }
    void on_publish(uint64_t key, uint32_t ver) {
        uint32_t d = done_[key].load(std::memory_order_acquire);
        if (ver < d) fail(c_->err_p3, key, -1, MISS, "P3: published v" + std::to_string(ver) + " after PUT v" + std::to_string(d) + " completed");
    }
    uint64_t errors() const { return c_->err_p1 + c_->err_p2 + c_->err_p3 + c_->err_bytes + c_->err_stale; }
    std::string summary() const {
        char b[256];
        snprintf(b, sizeof b, "gets_checked=%lu puts_done=%lu err_p1=%lu err_p2=%lu err_p3=%lu err_bytes=%lu%s%s",
                 (unsigned long)c_->gets.load(), (unsigned long)c_->puts.load(), (unsigned long)c_->err_p1.load(),
                 (unsigned long)c_->err_p2.load(), (unsigned long)c_->err_p3.load(), (unsigned long)c_->err_bytes.load(),
                 stale_p_ ? (" err_stale=" + std::to_string(c_->err_stale.load())).c_str() : "",
                 c_->absent ? (" absent=" + std::to_string(c_->absent.load())).c_str() : "");   // only once a GET found a delete
        return b;
    }
    std::vector<std::string> first_errors() {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<std::string> v = msgs_;
        v.insert(v.end(), late_.begin(), late_.end());
        return v;
    }

private:
    bool fail(std::atomic<uint64_t> &ctr, uint64_t key, int host, Kind kind, const std::string &what) {
        ctr++;
        std::lock_guard<std::mutex> g(mu_);
        static const char *kn[] = {"FAST", "SLOW", "MISS"};
        if (msgs_.size() < 20) {
            msgs_.push_back("key=" + std::to_string(key) + " host=" + std::to_string(host) + " path=" + kn[kind] + ": " + what);
        } else if (&ctr != &c_->err_p2 && late_.size() < 20) {
            // diagnosis: once the first 20 are taken (in committed runs: raw misses, P2, not counted), keep up to 20
            // more of the other kinds, so a counted error is never without its message
            late_.push_back("key=" + std::to_string(key) + " host=" + std::to_string(host) + " path=" + kn[kind] + ": " + what);
        }
        return false;
    }
    std::unique_ptr<std::atomic<uint32_t>[]> own_done_;
    std::atomic<uint32_t> *done_;
    std::unique_ptr<std::atomic<uint64_t>[]> own_cuts_;
    std::unique_ptr<std::atomic<uint32_t>[]> own_floor_;
    std::atomic<uint64_t> *cuts_ = nullptr;
    std::atomic<uint32_t> *floor_ = nullptr;
    uint64_t stale_p_ = 0, stale_slack_ = 0;
    CheckCounters own_cnt_, *c_;
    SimBackend *be_;
    uint64_t nkeys_ = 0;
    std::vector<std::unique_ptr<std::atomic<uint32_t>[]>> hwm_;   // P6: per host, per key
    std::mutex mu_;
    std::vector<std::string> msgs_, late_;
};

}  // namespace rc
