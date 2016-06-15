#!/bin/sh
## run_all_benchmark.sh for  in /Users/pouchet
##
## Made by Louis-Noel Pouchet
## Contact: <pouchet@cse.ohio-state.edu>
##

COMPILER="gcc -O3";

if ! [ -f "./benchmark_list" ]; then
    echo "[ERROR] This script should be run from the utilities/ directory.";
    exit 1;
fi;

POLYBENCH_ROOT=`pwd`/../;

for i in `cat benchmark_list`; do
    echo "Running $i";
    $COMPILER "$POLYBENCH_ROOT$i" -I $POLYBENCH_ROOT`dirname "$i"` -I . polybench.c -DPOLYBENCH_TIME -DSMALL_DATASET;
    ./time_benchmark.sh ./a.out;
    rm -f ./a.out;
done
