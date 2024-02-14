#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda.h>
#include <iostream>

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
static bool checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

#define TIMER_CREATE cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
#define TIMER_START cudaEventRecord(start, 0);
#define TIMER_STOP cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
// Get time elapsed in milliseconds
#define TIME_ELAPSED(var) float var=0; cudaEventElapsedTime(&var, start, stop);

#endif // UTILS_CUH