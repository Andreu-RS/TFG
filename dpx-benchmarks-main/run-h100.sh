echo "RUNNING BENCHMARKS FOR H100"
echo "============================"
echo "NO DPX UNROLL=1 BENCH_RUNS=3"
make run-no-dpx UNROLL=1 BENCH_RUNS=3
echo "============================"
echo "NO DPX UNROLL=2 BENCH_RUNS=3"
make run-no-dpx UNROLL=2 BENCH_RUNS=3
echo "============================"
echo "NO-DPX UNROLL=5 BENCH_RUNS=3"
make run-no-dpx UNROLL=5 BENCH_RUNS=3
echo "============================"
echo "NO-DPX UNROLL=10 BENCH_RUNS=3"
make run-no-dpx UNROLL=10 BENCH_RUNS=3
echo "============================"
echo "NO-DPX UNROLL=50 BENCH_RUNS=3"
make run-no-dpx UNROLL=50 BENCH_RUNS=3
echo "============================"
echo "NO-DPX UNROLL=100 BENCH_RUNS=3"
make run-no-dpx UNROLL=100 BENCH_RUNS=3
echo "============================"
echo "DPX UNROLL=1 BENCH_RUNS=3"
make run-dpx UNROLL=1 BENCH_RUNS=3
echo "============================"
echo "DPX UNROLL=2 BENCH_RUNS=3"
make run-dpx UNROLL=2 BENCH_RUNS=3
echo "============================"
echo "DPX UNROLL=5 BENCH_RUNS=3"
make run-dpx UNROLL=5 BENCH_RUNS=3
echo "============================"
echo "DPX UNROLL=10 BENCH_RUNS=3"
make run-dpx UNROLL=10 BENCH_RUNS=3
echo "============================"
echo "DPX UNROLL=50 BENCH_RUNS=3"
make run-dpx UNROLL=50 BENCH_RUNS=3
echo "============================"
echo "DPX UNROLL=100 BENCH_RUNS=3"
make run-dpx UNROLL=100 BENCH_RUNS=3
echo "============================"
