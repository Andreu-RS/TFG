#include "NW.cuh"

using namespace nw; // NW.cuh

//--NW alignment kernel implementation------------------------------------------
__global__ void nw::nw_align_gpu(
    DataStruct sequences, 
    Penalties penalties,
    MatrixesData matrixes, 
    Results results) 
{
    //--Set up------------------------------------------------------------------
    /*  NOTE:
        - In this simple version we will perform 1 aligment per thread.
        - NO PERFORMANCE IMPROVEMENTS have been considered.
        - The for() loop will distribute workload and avoid memory faults.
    */
    const uint32_t threadID = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t totalThreads = gridDim.x * blockDim.x;
    const uint32_t nAlignments = results.nAlignments;

    for(uint32_t alIdx = threadID; alIdx < nAlignments; alIdx += totalThreads) {
    //--Algorithm implementation------------------------------------------------
        // Parameters
        /*  NOTE:
            - We have only 1 target and 1 query and perform the alignment 
              several times.
            - For several pairs: 
                uint64_t tStartRef = sequences.targetRefs[alIdx];
                uint32_t tLen = sequences.targetRefs[alIdx+1] - tStartRef;
        */
        uint64_t tStartRef = sequences.targetRefs[0];
        uint64_t qStartRef = sequences.queryRefs[0];

        uint32_t tLen = sequences.targetRefs[1] - tStartRef;
        uint32_t qLen = sequences.queryRefs[1] - qStartRef;

        // Matrix initialization (Edge cases, auxiliar, not information)
        /*  NOTE:
			- We have 1 extra row and column for the initialization condition.
    	*/
        int16_t** scoreMatrix = &(matrixes.matrixes[matrixes.mRefs[alIdx]]);

        scoreMatrix[0][0] = 0;

        for(uint32_t colIdx = 1; colIdx <= tLen; ++colIdx) {
            scoreMatrix[0][colIdx] = scoreMatrix[0][colIdx-1] + penalties.insertion;
        }

        for(uint32_t rowIdx = 1; rowIdx <= qLen; ++rowIdx) {
            scoreMatrix[rowIdx][0] = scoreMatrix[rowIdx-1][0] + penalties.deletion;
        }

        // Compute scoreMatrix
        for (uint32_t rowIdx = 1; rowIdx <= qLen; ++rowIdx) {
            for (uint32_t colIdx = 1; colIdx <= tLen; ++colIdx) {

                int16_t auxPenalty = (sequences.targets[tStartRef+colIdx-1] == 
                    sequences.querys[qStartRef+rowIdx-1]) ? penalties.match : penalties.mismatch;

                scoreMatrix[rowIdx][colIdx] = __vimin3_s32(
                    scoreMatrix[rowIdx-1][colIdx-1] + auxPenalty,
                    scoreMatrix[rowIdx][colIdx-1] + penalties.insertion,
                    scoreMatrix[rowIdx-1][colIdx] + penalties.deletion);
            }
        }

        // Save result (bottom-right cell) to retrieve them to CPU
        storeResult(&results, alIdx, scoreMatrix[qLen][tLen]);
    }
}