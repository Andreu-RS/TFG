#include "DTW.cuh"

using namespace dtw; // DTW.cuh

//--NW alignment kernel implementation------------------------------------------
__global__ void dtw::dtw_align_gpu(
    DataStruct sequences,
    MatrixesData matrixes,  
    Results results) 
{
    //--Work distribution-------------------------------------------------------
    /*  NOTE:
        - In this simple version we will perform 1 aligment per thread.
        - NO PERFORMANCE IMPROVEMENTS have been considered.
        - The for() loop will distribute workload and avoid memory faults.
    */
    const uint32_t threadID = (blockIdx.x * blockDim.x) + threadIdx.x;
    const uint32_t totalThreads = gridDim.x * blockDim.x;
    const uint32_t nAlignments = results.nAlignments;

    for(uint32_t alIdx = threadID; alIdx < nAlignments; alIdx += totalThreads) {
    //--Environment set up------------------------------------------------------
        // Parameters
        /*  NOTE:
            - We have only 1 target and 1 query and perform the alignment 
              several times.
            - For several pairs: 
                uint64_t tStartRef = sequences.targetRefs[alIdx];
                uint32_t tLen = sequences.targetRefs[alIdx+1] - tStartRef;
        */
        uint32_t tStartRef = sequences.targetRefs[0];
        uint32_t qStartRef = sequences.queryRefs[0];

        uint32_t tLen = sequences.targetRefs[1] - tStartRef;
        uint32_t qLen = sequences.queryRefs[1] - qStartRef;

        // Matrix initialization (Edge cases, auxiliar, not information)
        /*  NOTE:
			- We have 1 extra row and column for the initialization condition.
    	*/
        uint32_t** costMatrix = &(matrixes.matrixes[matrixes.mRefs[alIdx]]);

        costMatrix[0][0] = 0;

        for(uint32_t colIdx = 1; colIdx <= tLen; colIdx++) {
            costMatrix[0][colIdx] = UINT32_MAX;
        }

        for(uint32_t rowIdx = 1; rowIdx <= qLen; rowIdx++) {
            costMatrix[rowIdx][0] = UINT32_MAX;
        }

    //--Algorithm implementation------------------------------------------------
        // Compute costMatrix
        for (uint32_t rowIdx = 1; rowIdx <= qLen; rowIdx++) {
            for (uint32_t colIdx = 1; colIdx <= tLen; colIdx++) {
                costMatrix[rowIdx][colIdx] = 
                    ABSDIFF(
                        sequences.targets[tStartRef+colIdx-1],
                        sequences.querys[qStartRef+rowIdx-1]) 
                    + __vimin3_u32(
                        costMatrix[rowIdx-1][colIdx-1],
                        costMatrix[rowIdx][colIdx-1],
                        costMatrix[rowIdx-1][colIdx]);
            }
        }

        // Save result (min last row) to retrieve them to CPU
        uint32_t result = costMatrix[qLen][tLen];
        for(uint32_t colIdx = 1; colIdx < tLen; ++colIdx) {
            result = min(result, costMatrix[qLen][colIdx]);
        }
        storeResult(&results, alIdx, result);
    }
}