#!/usr/bin/env bash

make

declare -A sizeLookup

function usage {
    echo "Usage: $0 [xs|s|m|l|xl]"
    exit
}

sizeLookup=( ["xs"]=5 ["s"]=6 ["m"]=7 ["l"]=8 ["xl"]=9 )

[[ "$#" -ne 1 ]] && usage
lookup="${sizeLookup["$1"]}"
[[ -z "$lookup" ]] && usage

grep '[0-9]' common/polybench.spec | sort | cut -f1,$lookup | sed 's|\t| |g;s|^|./dist/PolyBench/bin/|' | while read cmd
do
    echo $cmd
    $cmd
    echo ""
done
