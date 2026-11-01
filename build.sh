#!/bin/bash
# Build everything: the prototype (proto/build.sh -> proto/build-$(hostname)/), the H5 check and the read-path benchmark
# (bench/h5), the latency probe (bench/chase) and the YCSB trace generator (tools/ycsb), the last three into build/.
# usage: ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
export LC_ALL=C
bash proto/build.sh
mkdir -p build
CC=${CC:-gcc}; CXX=${CXX:-g++}
$CC -O2 -march=native -Wall -pthread -o build/h5_update bench/h5/update.c -lnuma
$CC -O2 -march=native -Wall -o build/chase bench/chase/chase.c -lnuma
$CXX -O2 -std=c++17 -Wall -o build/ycsb_gen tools/ycsb/ycsb_gen.cpp
echo "built build/{h5_update,chase,ycsb_gen}"
