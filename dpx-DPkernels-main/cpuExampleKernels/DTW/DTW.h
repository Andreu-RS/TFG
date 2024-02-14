#pragma 

#include <vector>
#include <iostream>
#include <stdint.h>

// Definitions
#define MIN(a,b) (((a)<=(b))?(a):(b))
#define MAX(a,b) (((a)>=(b))?(a):(b))
#define ABS(a,b) (((a)>=(b))?(a-b):(b-a))

void dtw_align(
    const std::vector<int> query,
    const uint16_t queryLength,
    const std::vector<int> target,
    const uint16_t targetLength,
    int** matrix,
    int* const score);

void dtw_out(
    int nAligments,
    std::vector<int> query,
    uint16_t queryLength,
    std::vector<int> target,
    uint16_t targetLength,
    int** matrix,
    int* score);