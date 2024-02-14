#ifndef BENCH_CUH
#define BENCH_CUH

#define BENCH_WRAPPER __attribute__((flatten))

#include <chrono>
#include <thread>
#include <array>
#include <functional>
#include <iostream>

#include "kernels.cuh"
#include "utils.cuh"
#include "logger.cuh"

namespace dpxbench {

    //-- Values definitions ----------------------------------------------------
    #ifndef DPX_BENCH_RUNS
    const int BENCH_RUNS = 1;
    #else
    const int BENCH_RUNS = DPX_BENCH_RUNS;
    #endif
    constexpr int NUM_BENCHMARKS = 40; // 36 DPX + 2 DP kernels
    // TODO
    int NUM_BLOCKS = 1000;
    int NUM_THREADS = 1024;

    //-- DPX wrappers ----------------------------------------------------------
    //-- Pure max/min --
    BENCH_WRAPPER
    void __vimax3_s32_bench_wrapper(int* values) {
        kernels::__vimax3_s32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vimin3_s32_bench_wrapper(int* values) {
        kernels::__vimin3_s32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vimax_s32_relu_bench_wrapper(int* values) {
        kernels::__vimax_s32_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vimin_s32_relu_bench_wrapper(int* values) {
        kernels::__vimin_s32_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vimax3_s32_relu_bench_wrapper(int* values) {
        kernels::__vimax3_s32_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vimin3_s32_relu_bench_wrapper(int* values) {
        kernels::__vimin3_s32_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vimax3_u32_bench_wrapper(int* values) {
        kernels::__vimax3_u32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vimin3_u32_bench_wrapper(int* values) {
        kernels::__vimin3_u32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    //-- Pure max/min 16x2 --
    BENCH_WRAPPER
    void __vimax3_s16x2_bench_wrapper(int* values) {
        kernels::__vimax3_s16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vimin3_s16x2_bench_wrapper(int* values) {
        kernels::__vimin3_s16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vimax_s16x2_relu_bench_wrapper(int* values) {
        kernels::__vimax_s16x2_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vimin_s16x2_relu_bench_wrapper(int* values) {
        kernels::__vimin_s16x2_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vimax3_s16x2_relu_bench_wrapper(int* values) {
        kernels::__vimax3_s16x2_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vimin3_s16x2_relu_bench_wrapper(int* values) {
        kernels::__vimin3_s16x2_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vimax3_u16x2_bench_wrapper(int* values) {
        kernels::__vimax3_u16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vimin3_u16x2_bench_wrapper(int* values) {
        kernels::__vimin3_u16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    //-- vib --
    BENCH_WRAPPER
    void __vibmax_s32_bench_wrapper(int* values) {
        kernels::__vibmax_s32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vibmin_s32_bench_wrapper(int* values) {
        kernels::__vibmin_s32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vibmax_u32_bench_wrapper(int* values) {
        kernels::__vibmax_u32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vibmin_u32_bench_wrapper(int* values) {
        kernels::__vibmin_u32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    //-- vib 16x2 --
    BENCH_WRAPPER
    void __vibmax_s16x2_bench_wrapper(int* values) {
        kernels::__vibmax_s16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vibmin_s16x2_bench_wrapper(int* values) {
        kernels::__vibmin_s16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __vibmax_u16x2_bench_wrapper(int* values) {
        kernels::__vibmax_u16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __vibmin_u16x2_bench_wrapper(int* values) {
        kernels::__vibmin_u16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    //-- add --
    BENCH_WRAPPER
    void __viaddmax_s32_bench_wrapper(int* values) {
        kernels::__viaddmax_s32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __viaddmin_s32_bench_wrapper(int* values) {
        kernels::__viaddmin_s32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __viaddmax_s32_relu_bench_wrapper(int* values) {
        kernels::__viaddmax_s32_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __viaddmin_s32_relu_bench_wrapper(int* values) {
        kernels::__viaddmin_s32_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __viaddmax_u32_bench_wrapper(int* values) {
        kernels::__viaddmax_u32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __viaddmin_u32_bench_wrapper(int* values) {
        kernels::__viaddmin_u32_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    //-- add 16x2 --
    BENCH_WRAPPER
    void __viaddmax_s16x2_bench_wrapper(int* values) {
        kernels::__viaddmax_s16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __viaddmin_s16x2_bench_wrapper(int* values) {
        kernels::__viaddmin_s16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __viaddmax_s16x2_relu_bench_wrapper(int* values) {
        kernels::__viaddmax_s16x2_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __viaddmin_s16x2_relu_bench_wrapper(int* values) {
        kernels::__viaddmin_s16x2_relu_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void __viaddmax_u16x2_bench_wrapper(int* values) {
        kernels::__viaddmax_u16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void __viaddmin_u16x2_bench_wrapper(int* values) {
        kernels::__viaddmin_u16x2_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    //-- DP kernels --
    BENCH_WRAPPER
    void NW_bench_wrapper(int* values) {
        kernels::NW_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void DTW_bench_wrapper(int* values) {
        kernels::DTW_bench<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    BENCH_WRAPPER
    void NW_bench_wrapper_16x2(int* values) {
        kernels::NW_bench_16x2<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }
    
    BENCH_WRAPPER
    void DTW_bench_wrapper_16x2(int* values) {
        kernels::DTW_bench_16x2<<<NUM_BLOCKS, NUM_THREADS>>>(values);
    }

    //-- DPX wrappers array ----------------------------------------------------
    const std::array<std::function<void(int*)>, NUM_BENCHMARKS> bench_wrapper = {
        //-- Pure max/min --
        __vimax3_s32_bench_wrapper,
        __vimin3_s32_bench_wrapper,
        __vimax_s32_relu_bench_wrapper,
        __vimin_s32_relu_bench_wrapper,
        __vimax3_s32_relu_bench_wrapper,
        __vimin3_s32_relu_bench_wrapper,
        __vimax3_u32_bench_wrapper,
        __vimin3_u32_bench_wrapper,

        //-- Pure max/min 16x2 ---
        __vimax3_s16x2_bench_wrapper,
        __vimin3_s16x2_bench_wrapper,
        __vimax_s16x2_relu_bench_wrapper,
        __vimin_s16x2_relu_bench_wrapper,
        __vimax3_s16x2_relu_bench_wrapper,
        __vimin3_s16x2_relu_bench_wrapper,
        __vimax3_u16x2_bench_wrapper,
        __vimin3_u16x2_bench_wrapper,
        
        //-- vib --
        __vibmax_s32_bench_wrapper,
        __vibmin_s32_bench_wrapper,
        __vibmax_u32_bench_wrapper,
        __vibmin_u32_bench_wrapper,

        //-- vib 16x2 --
        __vibmax_s16x2_bench_wrapper,
        __vibmin_s16x2_bench_wrapper,
        __vibmax_u16x2_bench_wrapper,
        __vibmin_u16x2_bench_wrapper,

        //-- add --
        __viaddmax_s32_bench_wrapper,
        __viaddmin_s32_bench_wrapper,
        __viaddmax_s32_relu_bench_wrapper,
        __viaddmin_s32_relu_bench_wrapper,
        __viaddmax_u32_bench_wrapper,
        __viaddmin_u32_bench_wrapper,

        //-- add 16x2 --
        __viaddmax_s16x2_bench_wrapper,
        __viaddmin_s16x2_bench_wrapper,
        __viaddmax_s16x2_relu_bench_wrapper,
        __viaddmin_s16x2_relu_bench_wrapper,
        __viaddmax_u16x2_bench_wrapper,
        __viaddmin_u16x2_bench_wrapper,

        //-- DP kernels --
        NW_bench_wrapper,
        DTW_bench_wrapper,
        NW_bench_wrapper_16x2,
        DTW_bench_wrapper_16x2
    };


    const std::array<std::string, NUM_BENCHMARKS> bench_names = {
        //-- Pure max/min --
        "__vimax3_s32",
        "__vimin3_s32",
        "__vimax_s32_relu",
        "__vimin_s32_relu",
        "__vimax3_s32_relu",
        "__vimin3_s32_relu",
        "__vimax3_u32",
        "__vimin3_u32",

        //-- Pure max/min 16x2 ---
        "__vimax3_s16x2",
        "__vimin3_s16x2",
        "__vimax_s16x2_relu",
        "__vimin_s16x2_relu",
        "__vimax3_s16x2_relu",
        "__vimin3_s16x2_relu",
        "__vimax3_u16x2",
        "__vimin3_u16x2",
        
        //-- vib --
        "__vibmax_s32",
        "__vibmin_s32",
        "__vibmax_u32",
        "__vibmin_u32",

        //-- vib 16x2 --
        "__vibmax_s16x2",
        "__vibmin_s16x2",
        "__vibmax_u16x2",
        "__vibmin_u16x2",

        //-- add --
        "__viaddmax_s32",
        "__viaddmin_s32",
        "__viaddmax_s32_relu",
        "__viaddmin_s32_relu",
        "__viaddmax_u32",
        "__viaddmin_u32",

        //-- add 16x2 --
        "__viaddmax_s16x2",
        "__viaddmin_s16x2",
        "__viaddmax_s16x2_relu",
        "__viaddmin_s16x2_relu",
        "__viaddmax_u16x2",
        "__viaddmin_u16x2",

        //-- DP kernels --
        "NW_kernel",
        "DTW_kernel",
        "NW_kernel (16x2)",
        "DTW_kernel (16x2)",
    };

    std::array<std::array<float, BENCH_RUNS>, bench_wrapper.size()> bench_results;

    //-- Bench function (entry point) ------------------------------------------
    void run_bench() {
        //std::cout << "\nRunning bench..." << std::endl;
        #ifndef NO_PROGRESS_BAR
        const int total_benchs = bench_wrapper.size() * BENCH_RUNS;
        #endif
        for (int dpxIdx=0; dpxIdx < bench_wrapper.size(); ++dpxIdx) {

            // Some mem allocation, just used to ensure kernels are performed
            int* values;
            cudaMalloc(&values, 10);
            CHECK_LAST_CUDA_ERROR();

            // Warmup call
            bench_wrapper[dpxIdx](values);

            cudaDeviceSynchronize();
            // Sleep 2 seconds to let the GPU cool down
            std::this_thread::sleep_for(std::chrono::seconds(2));

            for (int runIdx = 0; runIdx < BENCH_RUNS; ++runIdx) {
                TIMER_CREATE;
                TIMER_START;
                bench_wrapper[dpxIdx](values);
                TIMER_STOP;
                TIME_ELAPSED(telapsed);
                bench_results[dpxIdx][runIdx] = telapsed;

                CHECK_LAST_CUDA_ERROR();

                UPDATE_PROGRESS_BAR("Benchmarking", bench_names[dpxIdx], dpxIdx*BENCH_RUNS + runIdx, total_benchs)

                // Sleep 2 seconds to let the GPU cool down
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
        END_PROGRESS_BAR();
    }
} // namespace bench

#endif // BENCH_CUH