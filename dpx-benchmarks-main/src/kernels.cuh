#ifndef KERNELS_CUH
#define KERNELS_CUH

#define ABS(a,b) (((a)>=(b))?(a-b):(b-a))

#include <iostream>

namespace dpxbench {
namespace kernels {

constexpr long long int N = 2000000;

// Default unroll value if none is passed through the compiler
#ifndef DPXBENCH_UNROLL_NUM
constexpr int DPXBENCH_UNROLL = 1;
#else
constexpr int DPXBENCH_UNROLL = DPXBENCH_UNROLL_NUM;
#endif

//-- Pure max/min --
__global__ void __vimax3_s32_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimax3_s32(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimin3_s32_bench(int* v) {     
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimin3_s32(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimax_s32_relu_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<(N/2); i++) {
        a = __vimax_s32_relu(a, b);
        b = __vimax_s32_relu(a, c);
    }

    v[0] = a;
}

__global__ void __vimin_s32_relu_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<(N/2); i++) {
        a = __vimin_s32_relu(a, b);
        b = __vimin_s32_relu(a, c);
    }

    v[0] = a;
}

__global__ void __vimax3_s32_relu_bench(int* v) {  
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimax3_s32_relu(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimin3_s32_relu_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimin3_s32_relu(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimax3_u32_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimax3_u32(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimin3_u32_bench(int* v) {  
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimin3_u32(a, b, c);
    }

    v[0] = a;
}

//-- Pure max/min 16x2 --
__global__ void __vimax3_s16x2_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimax3_s16x2(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimin3_s16x2_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimin3_s16x2(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimax_s16x2_relu_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<(N/2); i++) { 
        a = __vimax_s16x2_relu(a, b);
        b = __vimax_s16x2_relu(a, c);
    }

    v[0] = a;
}

__global__ void __vimin_s16x2_relu_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<(N/2); i++) { 
        a = __vimin_s16x2_relu(a, b);
        b = __vimin_s16x2_relu(a, c);
    }

    v[0] = a;
}

__global__ void __vimax3_s16x2_relu_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimax3_s16x2_relu(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimin3_s16x2_relu_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimin3_s16x2_relu(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimax3_u16x2_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimax3_u16x2(a, b, c);
    }

    v[0] = a;
}

__global__ void __vimin3_u16x2_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __vimin3_u16x2(a, b, c);
    }

    v[0] = a;
}

//-- vib --
__global__ void __vibmax_s32_bench(int* v) {
    int a = v[0];
    int b = v[1];
    bool value;

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        int c = __vibmax_s32(a, b, &value);
    }

    v[0] = a;
}

__global__ void __vibmin_s32_bench(int* v) {
    int a = v[0];
    int b = v[1];
    bool value;

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        int c = __vibmin_s32(a, b, &value);
    }

    v[0] = a;
}

__global__ void __vibmax_u32_bench(int* v) {
    int a = v[0];
    int b = v[1];
    bool value;

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        int c = __vibmax_u32(a, b, &value);
    }

    v[0] = a;
}

__global__ void __vibmin_u32_bench(int* v) {
    int a = v[0];
    int b = v[1];
    bool value;

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        int c = __vibmin_u32(a, b, &value);
    }

    v[0] = a;
}

//-- vib 16x2 --
__global__ void __vibmax_s16x2_bench(int* v) {
    int a = v[0];
    int b = v[1];
    bool value;
    bool value2;

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        int c = __vibmax_s16x2(a, b, &value, &value2);
    }

    v[0] = a;
}

__global__ void __vibmin_s16x2_bench(int* v) {
    int a = v[0];
    int b = v[1];
    bool value;
    bool value2;

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        int c = __vibmin_s16x2(a, b, &value, &value2);
    }

    v[0] = a;
}

__global__ void __vibmax_u16x2_bench(int* v) {
    int a = v[0];
    int b = v[1];
    bool value;
    bool value2;

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        int c = __vibmax_u16x2(a, b, &value, &value2);
    }

    v[0] = a;
}

__global__ void __vibmin_u16x2_bench(int* v) {
    int a = v[0];
    int b = v[1];
    bool value;
    bool value2;

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        int c = __vibmin_u16x2(a, b, &value, &value2);
    }

    v[0] = a;
}

//-- add --
__global__ void __viaddmax_s32_bench(int* v) { 
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmax_s32(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmin_s32_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmin_s32(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmax_s32_relu_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmax_s32_relu(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmin_s32_relu_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmin_s32_relu(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmax_u32_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmax_u32(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmin_u32_bench(int* v) {
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmin_u32(a, b, c);
    }

    v[0] = a;
}

//-- add 16x2 --
__global__ void __viaddmax_s16x2_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmax_s16x2(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmin_s16x2_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmin_s16x2(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmax_s16x2_relu_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmax_s16x2_relu(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmin_s16x2_relu_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmin_s16x2_relu(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmax_u16x2_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmax_u16x2(a, b, c);
    }

    v[0] = a;
}

__global__ void __viaddmin_u16x2_bench(int* v) {   
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        a = __viaddmin_u16x2(a, b, c);
    }

    v[0] = a;
}

//-- DP kernels --
__global__ void NW_bench(int* v) {  
    // Define values (vector size is 10) 
    int a = v[0];
    int b = v[1];
    int c = v[2];
    int d = v[3];
    int e = v[4];
    int f = v[5];
    int g = v[6];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        // Algorithm here
        a = __vimin3_s32(
            b + ((a == c) ? d : e),
            a + f,
            a + g);
    }

    v[0] = a;
}

__global__ void DTW_bench(int* v) {   
    // Define values (vector size is 10)
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        // Algorithm here
        a = ABS(a, b) + __vimin3_s32(a, b, c);
    }

    v[0] = a;
}

__global__ void NW_bench_16x2(int* v) {  
    // Define values (vector size is 10) 
    int a = v[0];
    int b = v[1];
    int c = v[2];
    int d = v[3];
    int e = v[4];
    int f = v[5];
    int g = v[6];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        // Algorithm here
        a = __vimin3_s16x2(
            b + ((a == c) ? d : e),
            a + f,
            a + g);
    }

    v[0] = a;
}

__global__ void DTW_bench_16x2(int* v) {   
    // Define values (vector size is 10)
    int a = v[0];
    int b = v[1];
    int c = v[2];

    #pragma unroll(DPXBENCH_UNROLL)
    for (long long int i=0; i<N; i++) {
        // Algorithm here
        a = ABS(a, b) + __vimin3_s16x2(a, b, c);
    }

    v[0] = a;
}
 
} // namespace kernels
} // namespace dpxbench

#endif
