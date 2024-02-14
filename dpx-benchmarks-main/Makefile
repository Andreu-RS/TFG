NVCC=nvcc
NVCC_OPTIONS_DPX=-gencode arch=compute_90,code=sm_90
NVCC_OPTIONS_NO_DPX=-gencode arch=compute_80,code=sm_90

# Use this option if the output is being redirected to a file
NVCC_OPTIONS=-DNO_PROGRESS_BAR

UNROLL=1
NVCC_OPTIONS+=-DDPXBENCH_UNROLL_NUM=$(UNROLL)

BENCH_RUNS=1
NVCC_OPTIONS+=-DDPX_BENCH_RUNS=$(BENCH_RUNS)

# Generate the output as a python variable for easy processing
#NVCC_OPTIONS+=-DOUTPUT_PYTHON

compile-dpx:
	$(NVCC) $(NVCC_OPTIONS) $(NVCC_OPTIONS_DPX) -o dpx-bench src/main.cu

compile-no-dpx:
	$(NVCC) $(NVCC_OPTIONS) $(NVCC_OPTIONS_NO_DPX) -o dpx-bench src/main.cu

clean:
	rm -f dpx-bench

run-no-dpx: compile-no-dpx
	taskset -c 0 ./dpx-bench
run-dpx: compile-dpx
	taskset -c 0 ./dpx-bench
