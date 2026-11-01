# Krater: prototype

This is the prototype of **Krater**, a rack-scale key-value cache on shared, non-coherent CXL memory, from our PVLDB
submission. Each key has an owner host that makes all of the key's writes; every host reads any cached object directly
from the shared memory through its own CPU cache. Generation-checked reads validate a hit against the reader's own
index, owner-ordered writes reuse space without waiting for readers, and a host-timed change feed (a log that every host
polls every P) bounds staleness, with read-after-write for writers that ask for it.

The same framework implements the designs the paper compares against, and a trace replayer and a correctness checker
that run all of them on the same traces.

## Layout

| Path | What |
| --- | --- |
| `proto/src/rackcache.h`, `rackcache.cc` | The cache: shared-area layout, owner (fills, PUTs, eviction, log), readers (local index, hit check, slow path, poller), message rings, and the compared designs (`Config::scheme`) |
| `proto/src/sim.h` | Simulated backend (versioned store with an injected latency) and the checker (properties P1-P6) |
| `proto/src/replay.cc` | Trace replayer: one process (or thread group) per emulated host, closed or open loop, per-host statistics |
| `proto/src/stress.cc` | Randomized stress test with the checker, fault injection (stop / kill a host) |
| `proto/src/tests.cc` | Targeted tests: forced interleavings of the protocol and of each ablation's counterexample |
| `proto/src/hitbench.cc`, `robench.cc`, `roflush.cc` | Micro-benchmarks of the hit path and of read-only mappings |
| `proto/build.sh` | Builds the above into `proto/build-$(hostname)/` |
| `bench/h5/` | The H5 check (`update.c`, `run_h5.sh`), see below |
| `bench/chase/chase.c` | Pointer-chasing load-latency probe (e.g. CXL node vs local DRAM) |
| `bench/rpc/rpc_dist.h` | Header used by the sharding baseline's injected RPC latency |
| `tools/ycsb/` | YCSB-B / -C trace generator (`ycsb_gen`) in the replayer's binary trace format |
| `examples/run_minimal.sh` | Minimal end-to-end example |
| `build.sh` | Builds everything |

Designs (`--scheme` of `replay` / `stress`, as named in the paper):

| Paper | Flags |
| --- | --- |
| Krater | `--scheme rackcache` |
| XTRA-style | `--scheme xtra --xtra-own-freshness 1` (one directory; add `--xtra-dir-replicas 2` for a directory copy per reader) |
| Beluga-style | `--scheme cxl_uc` |
| Per-host copies | `--scheme priv` |
| SHARD-CXL | `--scheme shard --rpc-ns 1700 --shard-direct` (threads mode, no `--procs`) |
| MEGALON-style | `--scheme gstate_coh` (its state word in a coherent region); `--scheme gstate_cxl` reads it from non-coherent memory, flushing first |
| Invalidation | `--update invalidate` (`replay`, `stress`) |
| Delayed reuse | `--reuse delayed` (`stress`) |

PUT acknowledgement: `--put-order swap --put-wait 0` = committed (the default contract, stale at most P);
`--put-order window --put-wait 1` = read-after-write (the PUT returns W = P after its cut). P = W is `--poll-us`.

## Requirements

- Linux on x86-64, g++ with C++17 (we used g++ 11 and 14), and libnuma (`libnuma-dev` / `numactl-devel`).
- For measurements: a CPU with AVX-512F (an aligned 64-byte load per line; H5) and CLWB / CLFLUSHOPT. AVX-512BW is used
  for byte-masked stores when present. Without AVX-512F the build still works for functional tests (it prints a warning;
  a line is then read with two 32-byte loads, so H5 does not hold).
- The paper's setup: a 4-socket server whose CXL Type-3 memory expander appears as a CPU-less NUMA node. Each socket acts
  as one host (one process per socket, `--procs --bind`), and the shared area lives on the CXL node (`--shm-node <node>`).
  Any multi-socket machine runs everything functionally; with no CXL node, put the shared area on a remote DRAM node.

## Build

```bash
./build.sh            # proto/build-$(hostname)/{tests,stress,replay,hitbench,robench,roflush}, build/{h5_update,chase,ycsb_gen}
```

## Correctness

```bash
proto/build-$(hostname)/tests            # every suite, one process; exit 0 = every scenario as expected
proto/build-$(hostname)/tests --procs    # the same with one process per host (POSIX shared memory)
proto/build-$(hostname)/tests --suite log --emu   # one suite, with emulated non-coherent reader caches
proto/build-$(hostname)/stress --procs --hosts 4 --clients 4 --seconds 60 --shm-node 0   # randomized, checked; RESULT OK errors=0
```

`tests` forces each interleaving with blocking hooks; for every rule of the protocol it also runs the variant with the
rule turned off and expects the checker to report the counterexample (rows marked `info`). The full run takes tens of
minutes. `stress` accepts the same `--scheme` / `--update` / `--put-order` flags as `replay`, and `--emu` for emulated
non-coherence.

## Minimal example

```bash
./build.sh
examples/run_minimal.sh          # [SHM_NODE=n] [THREADS=2] [KEYS=100000] [REQUESTS=2000000]
```

It generates a small YCSB-B trace (95% reads, Zipf 0.99, 1 KB objects) and replays it on 4 emulated hosts with every
design: closed loop, 10 us backend, a cache of 15% of the keys, every GET checked against the backend's versions. Each run
prints its `RESULT` line (hit ratio, mean GET latency, `errors=0`); the per-host throughput and latency lines start with
`# host`. One replay by hand:

```bash
build/ycsb_gen --workload=B --keys=100000 --requests=2000000 --clients=64 /tmp/ycsb-b
proto/build-$(hostname)/replay --bin /tmp/ycsb-b/records.bin --objsize /tmp/ycsb-b/objects.bin --routing client \
  --pct 15 --scheme rackcache --perf --procs --bind --shm-node <cxl node> --threads 2 --delay-ns 10000 \
  --write-delay-ns 10000 --put-inflight 0 --poll-us 100 --heat 256 --accounting pool --li-cap 1 --feed 1 --kg cxl \
  --m-base objsize --warm-frac 0.1 --seed 1 --put-order swap --put-wait 0
```

Add `--lean` to time the cache alone: the clients then skip the per-request checks (the throughput experiments of the
paper run this way; the freshness and fault experiments do not). `replay` without arguments lists every option (open
loop `--open-rate`, `--hosts`, backend latency distributions, ...).

## H5 check

H5 is the hardware assumption that an aligned 64-byte load reads one instance of a cache line. `bench/h5/run_h5.sh` runs
an owner that rewrites objects in place (each line: generation first, then payload) while readers on the other sockets
read with one 64-byte load per line and check the generations, as Krater's hits do; every read that passes is verified
against the payload it must hold. A torn load would pass the check with a mixed payload and count as an undetected error.

```bash
bench/h5/run_h5.sh               # [DATA_NODE=n] [OWNER_CPU=c] [READER_CPUS=a,b,c] [SECS=5]; pass = 0 undetected errors
```

Defaults: data on the highest-numbered NUMA node (the CXL node if it is CPU-less), the owner on node 1, readers on up to
three other nodes. `build/h5_update ... <check_mode>` with check_mode 2 / 1 / 0 runs weaker checks as controls.
