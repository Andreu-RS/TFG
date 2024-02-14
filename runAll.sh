#!/bin/bash
echo "============================"
echo "============================"
echo "RUNNING MICROBENCHMARKS"
cd dpx-benchmarks-main
./run-h100.sh
cd ..
echo "============================"
echo "============================"
echo "RUNNING DP KERNELS"
cd dpx-DPkernels-main
./run-h100.sh
cd ..
echo "============================"
echo "============================"