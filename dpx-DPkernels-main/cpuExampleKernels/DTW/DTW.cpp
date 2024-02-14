#include "DTW.h"

void dtw_align( 
    const std::vector<int> query,
    const uint16_t queryLength,
    const std::vector<int> target,
    const uint16_t targetLength,
    int** costMatrix,
    int* const score)
{
    // Parameters
    uint16_t rowIdx;
    uint16_t colIdx;

    // Matrix Initialization (Edge cases, auxiliar, not information)
    costMatrix[0][0] = 0;
    for (colIdx = 1; colIdx <= targetLength; ++colIdx) {
        costMatrix[0][colIdx] = INT32_MAX;
    }

    for (rowIdx = 1; rowIdx <= queryLength; ++rowIdx) {
        costMatrix[rowIdx][0] = INT32_MAX;
    }

    // Compute scoreMatrix (except last row)
    for (rowIdx = 1; rowIdx < queryLength; ++rowIdx) {
        for (colIdx = 1; colIdx <= targetLength; ++colIdx) {
            int min = costMatrix[rowIdx-1][colIdx-1];
            min = MIN(min,costMatrix[rowIdx][colIdx-1]);
            min = MIN(min,costMatrix[rowIdx-1][colIdx]);
            costMatrix[rowIdx][colIdx] = ABS(query[rowIdx-1], target[colIdx-1]) + min;
        }
    }

    //compute last row and score
    *score = INT32_MAX;
    for (colIdx = 1; colIdx <= targetLength; ++colIdx) {
        int min = costMatrix[queryLength-1][colIdx-1];
        min = MIN(min,costMatrix[queryLength][colIdx-1]);
        min = MIN(min,costMatrix[queryLength-1][colIdx]);
        costMatrix[rowIdx][colIdx] = ABS(query[rowIdx-1], target[colIdx-1]) + min;
        *score = MIN(*score, costMatrix[rowIdx][colIdx]);
    }
}

void dtw_out(
    int alIdx,
    std::vector<int> query,
    uint16_t queryLength,
    std::vector<int> target,
    uint16_t targetLength,
    int** matrix,
    int* score)
{
    std::cout << "\nAlIdx: " << alIdx << "\nScore: " << *score 
        << std::endl;
    /*
    //TEST CODE START
        << "\nTarget: ";
        for(int i = 0; i < targetLength; i++) {
            std::cout << target[i] << " ";
        }
        std::cout << "\nQuery: ";
        for(int i = 0; i < queryLength; i++) {
            std::cout << query[i] << " ";
        }
        std::cout << "\nMatrix:\n";
        for(int i = 1; i <= queryLength; i++) {
            for(int j = 1; j <= targetLength; j++) {
                std::cout << "\t" << matrix[i][j];
            }
            std::cout << "\n";
        }
    //TEST CODE END
    */
}