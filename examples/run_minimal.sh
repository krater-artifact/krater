#!/bin/bash
# Minimal end-to-end example: generate a small YCSB-B trace (95% reads, 5% updates, Zipf 0.99, 1 KB objects) and replay it
# on 4 emulated hosts with every design, closed loop, a 10 us backend, cache = 15% of the key space, every GET checked
# against the backend's versions (non-lean). Each run ends with a "RESULT ... errors=0" line; the throughput and latency
# lines come before it. On a 4-socket machine with a CXL memory node, host h runs on socket h and the shared area lives on
# the CXL node (as in the paper); elsewhere host h runs on NUMA node h mod nodes, which checks function, not performance.
# usage: [SHM_NODE=n] [THREADS=2] [KEYS=100000] [REQUESTS=2000000] examples/run_minimal.sh [outdir]
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
B=$R/proto/build-$(hostname)
[ -x "$B/replay" ] && [ -x "$R/build/ycsb_gen" ] || { echo "build first: ./build.sh"; exit 2; }
O=${1:-$R/examples/out}; mkdir -p "$O"
nodes=$(ls -d /sys/devices/system/node/node[0-9]* | sed 's#.*node##' | sort -n)
cpuless=$(for n in $nodes; do [ -z "$(cat /sys/devices/system/node/node$n/cpulist)" ] && echo $n; done | tail -1)
SHM_NODE=${SHM_NODE:-${cpuless:-0}}
THREADS=${THREADS:-2}; KEYS=${KEYS:-100000}; REQUESTS=${REQUESTS:-2000000}
T=$O/trace-ycsb-b
[ -s "$T/records.bin" ] || "$R/build/ycsb_gen" --workload=B --sizes=ycsb --keys=$KEYS --requests=$REQUESTS --clients=64 "$T"
COMMON="--bin $T/records.bin --objsize $T/objects.bin --routing client --pct 15 --delay-ns 10000 --write-delay-ns 10000
  --heat 256 --accounting pool --seed 1 --warm-frac 0.1 --perf --bind --shm-node $SHM_NODE --put-inflight 0 --poll-us 100
  --li-cap 1 --feed 1 --kg cxl --m-base objsize --threads $THREADS --put-order swap --put-wait 0"
echo "# shared area on node $SHM_NODE, $THREADS client threads per host, trace $T"
for d in "krater:--procs --scheme rackcache" \
         "xtra-style:--procs --scheme xtra --xtra-own-freshness 1" \
         "xtra-style-xrep2:--procs --scheme xtra --xtra-own-freshness 1 --xtra-dir-replicas 2" \
         "beluga-style:--procs --scheme cxl_uc" \
         "per-host-copies:--procs --scheme priv" \
         "shard-cxl:--scheme shard --rpc-ns 1700 --shard-direct"; do
  n=${d%%:*}
  timeout 900 "$B/replay" $COMMON ${d#*:} > "$O/$n.out" 2> "$O/$n.err"; rc=$?
  printf '%-18s exit=%s  %s\n' "$n" "$rc" "$(grep -oE '^RESULT [a-z-]+ scheme=[a-z_]+|hit_ratio=[0-9.]+|get_ns_mean=[0-9.]+| errors=[0-9]+' "$O/$n.out" | tr '\n' ' ')"
done
echo "# per-host throughput: grep '^# host' $O/<design>.out ; add --lean to time the cache alone (no per-GET checks)"
