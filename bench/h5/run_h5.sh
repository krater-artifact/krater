#!/bin/bash
# H5 check (the paper's hardware assumption: an aligned 64-byte load reads one instance of the line). One owner thread
# rewrites objects in place on DATA_NODE (each line: its 4-byte generation first, then the 60-byte payload, write-back),
# while readers on other sockets read with ONE aligned 64-byte load per line and check every line's generation, as
# Krater's hits do. Every read that passes the check is verified against the payload it must hold; a mismatch is an
# UNDETECTED error. A 64-byte load that tore (old generation, new payload bytes) would show up as one. Pass = 0 undetected
# errors in every run (the program exits with 3 otherwise). Needs AVX-512F.
# usage: [DATA_NODE=n] [OWNER_CPU=c] [READER_CPUS=a,b,c] [SECS=5] bench/h5/run_h5.sh [build/h5_update]
#   defaults: DATA_NODE = the highest-numbered NUMA node (a CPU-less CXL node if there is one), OWNER_CPU = the first CPU of
#   node 1 (or 0), READER_CPUS = the first CPU of up to three other nodes (or CPUs 1-3 on a one-node machine).
set -uo pipefail
B=${1:-$(cd "$(dirname "$0")/../.." && pwd)/build/h5_update}
[ -x "$B" ] || { echo "build first: ./build.sh"; exit 2; }
first_cpu() { local l; l=$(cat /sys/devices/system/node/node$1/cpulist 2>/dev/null); l=${l%%,*}; echo "${l%%-*}"; }
nodes=$(ls -d /sys/devices/system/node/node[0-9]* | sed 's#.*node##' | sort -n)
cpu_nodes=$(for n in $nodes; do [ -n "$(cat /sys/devices/system/node/node$n/cpulist)" ] && echo $n; done)
DATA_NODE=${DATA_NODE:-$(echo "$nodes" | tail -1)}
on=$(echo "$cpu_nodes" | grep -qx 1 && echo 1 || echo "$cpu_nodes" | head -1)
OWNER_CPU=${OWNER_CPU:-$(first_cpu $on)}
if [ -z "${READER_CPUS:-}" ]; then
  READER_CPUS=$(for n in $cpu_nodes; do [ "$n" != "$on" ] && first_cpu $n; done | head -3 | paste -sd,)
  [ -n "$READER_CPUS" ] || READER_CPUS=$((OWNER_CPU + 1)),$((OWNER_CPU + 2)),$((OWNER_CPU + 3))
fi
SECS=${SECS:-5}
echo "# h5: data node $DATA_NODE, owner cpu $OWNER_CPU, reader cpus $READER_CPUS, $SECS s per run"
echo "check_mode,store_order,value_size,hotset,rate_target,rate_achieved,rep,reader,cpu,mops,ops,detections,stale,torn,retries,retry_ns_avg,undetected,ns_per_op,retry_ns_p50,retry_ns_p90,retry_ns_p99,retry_ns_max,retry_samples"
bad=0
for hot in $((64 << 10)) $((8 << 20)); do
  for vs in 256 4096; do
    for rate in 10000000 100000; do
      out=$("$B" $vs $hot $rate $SECS 0 $DATA_NODE 3 0 $OWNER_CPU $READER_CPUS 2>&1); rc=$?
      echo "$out" | grep '^CSV,' | sed 's/^CSV,//'
      [ $rc = 0 ] || { echo "# run vs=$vs hot=$hot rate=$rate: exit $rc"; echo "$out" | grep -v '^CSV,' | tail -3; bad=1; }
    done
  done
done
[ $bad = 0 ] && echo "# H5 check passed: 0 undetected errors" || { echo "# H5 check FAILED (see above)"; exit 3; }
