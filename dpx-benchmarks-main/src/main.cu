#include <iostream>

#include "bench.cuh"
#include "gpu.cuh"

int main(int argc, char** argv) {

    dpxbench::gpu::GPUs gpus;

    gpus.pprint();

    // Adjust blocks and threads per block to fill the GPU scheduler
    // TODO: This strategy can lead to allocating more warps per SM than the
    //       maximum that can be concurrently running.
    const int max_warps_per_sm = gpus.get_max_threads_per_multiprocessor() / 32;
    const int max_threads_per_block = gpus.get_max_threads_per_block();
    const int blocks_per_sm = ((max_warps_per_sm*32) / max_threads_per_block) + ((max_warps_per_sm*32) % max_threads_per_block != 0);

    dpxbench::NUM_BLOCKS = gpus.get_multi_processor_count() * blocks_per_sm;
    dpxbench::NUM_THREADS = max_threads_per_block;

    printf("Running benchmarks. Blocks: %d, Threads per block: %d, Runs: %d\n", dpxbench::NUM_BLOCKS, dpxbench::NUM_THREADS, dpxbench::BENCH_RUNS);

    dpxbench::run_bench();

    const double num_gops = (double)((size_t)dpxbench::NUM_BLOCKS * dpxbench::NUM_THREADS * dpxbench::BENCH_RUNS * dpxbench::kernels::N) / 1000000000L;

    #ifdef OUTPUT_PYTHON
        std::cout << "results_gpdpx = {\n";
    #endif

    for (int i=0; i<dpxbench::NUM_BENCHMARKS; i++) {
        float avg = 0;
        for (int j=0; j<dpxbench::BENCH_RUNS; j++) {
            avg += dpxbench::bench_results[i][j];
        }

        const double seconds = avg / 1000;
        double gopspersecond;
        if (dpxbench::bench_names[i] == "NW_kernel (16x2)") {
            gopspersecond = num_gops*2 / seconds;
        } else if (dpxbench::bench_names[i] == "DTW_kernel (16x2)") {
            gopspersecond = num_gops*2 / seconds;
        } else {
            gopspersecond = num_gops / seconds;
        }

        #ifdef OUTPUT_PYTHON
            std::cout << "'" << dpxbench::bench_names[i] << "': " << gopspersecond << "," << std::endl;
        #else
            std::cout << dpxbench::bench_names[i] << " " << seconds << " " << gopspersecond << std::endl;
        #endif
    }

    #ifdef OUTPUT_PYTHON
        std::cout << "}" << std::endl;
    #endif

    return 0;
}