#!/bin/bash
# Build the prototype (tests, stress, replay, hitbench, robench, roflush) into proto/build-$(hostname)/ (per host: the
# binaries use -march=native).
# usage: [RC_MONO=1] proto/build.sh [extra CXXFLAGS]
#   RC_MONO=1: compile in the optional monotonic-reads fix (Config::mono), into build-$(hostname)-mono1/; the default (0)
#   leaves it out of the GET / PUT paths (the measured build)
set -euo pipefail
export LC_ALL=C   # English compiler messages
SRC=$(cd "$(dirname "$0")" && pwd)/src
RC_MONO=${RC_MONO:-0}
SUF=""; if [ "$RC_MONO" = 1 ]; then SUF=-mono1; fi
OUT=$(cd "$(dirname "$0")" && pwd)/build-$(hostname)$SUF
mkdir -p "$OUT"
CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -O2 -g -march=native -Wall -Wextra -Wno-unused-parameter -pthread -DRC_MONO=$RC_MONO ${*:-}"
$CXX $CXXFLAGS -c "$SRC/rackcache.cc" -o "$OUT/rackcache.o"
$CXX $CXXFLAGS "$SRC/tests.cc" "$OUT/rackcache.o" -o "$OUT/tests" -lnuma
$CXX $CXXFLAGS "$SRC/stress.cc" "$OUT/rackcache.o" -o "$OUT/stress" -lnuma
$CXX $CXXFLAGS "$SRC/replay.cc" "$OUT/rackcache.o" -o "$OUT/replay" -lnuma
$CXX $CXXFLAGS "$SRC/hitbench.cc" "$OUT/rackcache.o" -o "$OUT/hitbench" -lnuma
$CXX $CXXFLAGS "$SRC/robench.cc" "$OUT/rackcache.o" -o "$OUT/robench" -lnuma
$CXX $CXXFLAGS "$SRC/roflush.cc" -o "$OUT/roflush" -lnuma
echo "built $OUT/{tests,stress,replay,hitbench,robench,roflush} with $($CXX --version | head -1)"
