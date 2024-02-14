#include "NW.h"

void nw_align(
    const Penalties penalties,
    const std::string query,
    const uint16_t queryLength,
    const std::string target,
    const uint16_t targetLength,
    int** scoreMatrix) 
{
    // Parameters
    uint16_t rowIdx;
    uint16_t colIdx;

    // Matrix Initialization (Edge cases, auxiliar, not information)
    scoreMatrix[0][0] = 0;
    for (colIdx = 1; colIdx <= targetLength; ++colIdx) {
        scoreMatrix[0][colIdx] = scoreMatrix[0][colIdx-1] + penalties.insertion;
    }

    for (rowIdx = 1; rowIdx <= queryLength; ++rowIdx) {
        scoreMatrix[rowIdx][0] = scoreMatrix[rowIdx-1][0] + penalties.deletion;
    }

    // Compute scoreMatrix
    for (rowIdx = 1; rowIdx <= queryLength; ++rowIdx) {
        for (colIdx = 1; colIdx <= targetLength; ++colIdx) {
            int min = scoreMatrix[rowIdx-1][colIdx-1] + 
                ((target[colIdx-1] == query[rowIdx-1]) ? penalties.match : penalties.mismatch);

            min = MIN(min,scoreMatrix[rowIdx][colIdx-1] + penalties.insertion);
            min = MIN(min,scoreMatrix[rowIdx-1][colIdx] + penalties.deletion);
            scoreMatrix[rowIdx][colIdx] = min;
        }
    }
}

void nw_out(
    int alIdx,
    std::string query,
    uint16_t queryLength,
    std::string target,
    uint16_t targetLength,
    int** matrix)
{
    std::cout << "\nAlIdx: " << alIdx << "\nScore: " << matrix[queryLength][targetLength]
        << std::endl;

    /*
    //TEST CODE START
        << "\nTarget: " << target << "\nQuery: " << query 
        << "\nMatrix:\n";
        for(int i = 1; i <= queryLength; i++) {
            for(int j = 1; j <= targetLength; j++) {
                std::cout << "\t" << matrix[i][j];
            }
            std::cout << "\n";
        }
    //TEST CODE END 
    */
}