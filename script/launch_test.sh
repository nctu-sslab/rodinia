#!/usr/bin/env bash

echo "Installing release version of omp runtime"

rm -f /tmp/AT_stats.txt
ninja -C $LLVM_BUILD_PATH/../build-Release-openmp install &> /dev/null
./verify-all.py
