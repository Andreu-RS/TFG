#pragma once
#ifndef DTW_CUH
#define DTW_CUH

#include <string.h>
#include <cstdint>
#include <cuda.h>
#include <iostream>

#include "../Utilities/utils.cuh"

namespace dtw {

//--Definitions-----------------------------------------------------------------
#define ABSDIFF(a,b) (((a)>=(b))?(a-b):(b-a))

//--Structures to hold data-----------------------------------------------------
/*	NOTE:
	- To ease the data transfer to the GPU we will save all sequences one after
	  another and their position.
	- Size of references is total+1 to allow compute length of last sequence. 
	  We set the first ref to 0 in initialization of struct.
	  When adding a sequence we will update the ref of the next one.
	  (See functions below)
*/
typedef struct{
    uint32_t* targets; // (target) Sequences (one after another)
	uint32_t* querys;
	uint32_t* targetRefs; // Position 1st element of each (target) sequence
	uint32_t* queryRefs;
	uint32_t nTargets; // Total (target) sequences
	uint32_t nQuerys;
} DataStruct;

/*	NOTE:
	- matrixes is an array of GPU pointers, each pointer references a row.
	- Each matrix has the EXACT amount of memory needed for the corresponding
	  alignment.
	- Each alignment needs a matrix of (querySize+1)*(targetSize+1).
*/
typedef struct{
    uint32_t** matrixes;
	uint32_t* mRefs;
	uint32_t nAlignments;
} MatrixesData;

typedef struct{
    uint32_t* results;
	uint32_t nAlignments;
} Results;


//--CPU functions---------------------------------------------------------------
inline void initializeDataStruct(DataStruct* data) {
	data->nTargets = 0;
	data->nQuerys = 0;
	
	data->targetRefs = (uint32_t*) malloc(2*sizeof(uint32_t));
	data->targetRefs[0] = 0;
	data->targets = NULL;

	data->queryRefs = (uint32_t*) malloc(2*sizeof(uint32_t));
	data->queryRefs[0] = 0;
	data->querys = NULL;
}

inline void storeTarget(uint32_t* sequence, size_t size, DataStruct* data) {
	// Update value (consider it for memory accesses)
	data->nTargets++;
	
	// Allocate and initialize the new reference (start of NEXT sequence)
	size_t refSize = (data->nTargets+1)*sizeof(uint32_t);
	data->targetRefs = (uint32_t*) realloc(data->targetRefs, refSize);
	data->targetRefs[data->nTargets] = data->targetRefs[data->nTargets-1]+size;

	// Allocate memory for the new sequence
	size_t tsTotalSize = (data->targetRefs[data->nTargets])*sizeof(uint32_t);
	data->targets = (uint32_t*) realloc(data->targets, tsTotalSize);

	// Copy data to struct
	size_t tSize = size*sizeof(uint32_t);
	memcpy(&(data->targets[data->targetRefs[data->nTargets-1]]), sequence, tSize);
}

inline void storeQuery(uint32_t* sequence, size_t size, DataStruct* data) {
	// Update value (consider it for memory accesses)
	data->nQuerys++;
	
	// Allocate and initialize the new reference (start of NEXT sequence)
	size_t refSize = (data->nQuerys+1)*sizeof(uint32_t);
	data->queryRefs = (uint32_t*) realloc(data->queryRefs, refSize);
	data->queryRefs[data->nQuerys] = data->queryRefs[data->nQuerys-1]+size;

	// Allocate memory for the new sequence
	size_t qsTotalSize = (data->queryRefs[data->nQuerys])*sizeof(uint32_t);
	data->querys = (uint32_t*) realloc(data->querys, qsTotalSize);

	// Copy data to the struct
	size_t qSize = size*sizeof(uint32_t);
	memcpy(&(data->querys[data->queryRefs[data->nQuerys-1]]), sequence, qSize);

}

inline void setMatrixesDataGPU(MatrixesData* matrixesStruct, DataStruct* data, uint32_t nAlignments) {
	// Auxiliars to ease redability and hold data
	/*  NOTE:
        - We need to add 1 extra row and column for the initialization condition
          which needs to be considered when accessing the matrix.
		- The total amount of rows will be the total amount of query characters
		  + the number of alignments (due to the extra row).
    */
	uint32_t totalRows = (data->queryRefs[data->nQuerys] + 1) * nAlignments;

	uint32_t** matrixesAux;
	uint32_t* mRefsAux;

	cudaMemcpyKind h2d = cudaMemcpyHostToDevice;

	// Set values in auxiliars
	size_t mTotalSize = totalRows*sizeof(uint32_t*);
	matrixesAux = (uint32_t**) malloc(mTotalSize);

	size_t mRefsSize = (nAlignments+1)*sizeof(uint32_t);
	mRefsAux = (uint32_t*) malloc(mRefsSize);
	mRefsAux[0] = 0;

	for(uint32_t alIdx = 0; alIdx < nAlignments; alIdx++) {
		//uint32_t nRows = data->queryRefs[alIdx+1] - data->queryRefs[alIdx] +1;
		//uint32_t nCols = data->targetRefs[alIdx+1] - data->targetRefs[alIdx] +1;

		uint32_t nRows = data->queryRefs[1] - data->queryRefs[0] +1;
		uint32_t nCols = data->targetRefs[1] - data->targetRefs[0] +1;
		size_t rSize = nCols*sizeof(uint32_t);

		mRefsAux[alIdx+1] = mRefsAux[alIdx] + nRows;

		for(uint32_t rowIdx = 0; rowIdx < nRows; rowIdx++) {
			//cudaMalloc(&(matrixesAux[mRefsAux[alIdx]+rowIdx]), rSize);
			cudaError err = cudaMalloc(&(matrixesAux[mRefsAux[alIdx]+rowIdx]), rSize);
			if(err != cudaSuccess){
				std::cout << "ERROR ALLOCATING GPU MEMORY: " << cudaGetErrorString(err) << std::endl;
				return;
			}
		}
	}

	// Set/copy values in GPU struct
	matrixesStruct->nAlignments = nAlignments;

	cudaMalloc(&(matrixesStruct->matrixes), mTotalSize);
	cudaMemcpy(matrixesStruct->matrixes, matrixesAux, mTotalSize, h2d);

	cudaMalloc(&(matrixesStruct->mRefs), mRefsSize);
	cudaMemcpy(matrixesStruct->mRefs, mRefsAux, mRefsSize, h2d);

	// Check for errors
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();

	// Free auxiliars
	free(matrixesAux);
	free(mRefsAux);
}

inline void initializeResultsGPU(Results* resultStruct, uint32_t nAlignments) {
	size_t resSize = nAlignments*sizeof(uint32_t);

	resultStruct->nAlignments = nAlignments;

	resultStruct->results = NULL;
	cudaMalloc(&(resultStruct->results), resSize);

	// Check for errors
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();
}


//--GPU functions---------------------------------------------------------------
inline __device__ void storeResult(Results* resultStruct, uint32_t alIdx, uint32_t result) {
	resultStruct->results[alIdx] = result;
}


//--CPU-GPU memory transfer functions-------------------------------------------
inline void copySequencesToGPU(DataStruct* dataCPU, DataStruct* dataGPU) {
	// Auxiliars to ease redability
	uint32_t nTargets = dataCPU->nTargets;
	uint32_t nQuerys = dataCPU->nQuerys;

	size_t tSize = dataCPU->targetRefs[nTargets]*sizeof(uint32_t);
	size_t qSize = dataCPU->queryRefs[nQuerys]*sizeof(uint32_t);

	size_t tRefSize = (nTargets +1)*sizeof(uint32_t);
	size_t qRefSize = (nQuerys +1)*sizeof(uint32_t);

	cudaMemcpyKind h2d = cudaMemcpyHostToDevice;

	// Reference pointers
	dataGPU->targets = NULL;
	dataGPU->querys = NULL;
	dataGPU->targetRefs = NULL;
	dataGPU->queryRefs = NULL;

	// Allocate and copy sequences
	cudaMalloc(&(dataGPU->targets), tSize);
	cudaMemcpy(dataGPU->targets, dataCPU->targets, tSize, h2d);

	cudaMalloc(&(dataGPU->querys), qSize);
	cudaMemcpy(dataGPU->querys, dataCPU->querys, qSize, h2d);

	// Allocate and copy references
	cudaMalloc(&(dataGPU->targetRefs), tRefSize);
	cudaMemcpy(dataGPU->targetRefs, dataCPU->targetRefs, tRefSize, h2d);

	cudaMalloc(&(dataGPU->queryRefs), qRefSize);
	cudaMemcpy(dataGPU->queryRefs, dataCPU->queryRefs, qRefSize, h2d);

	// Check for errors
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();
}

inline void copyResultsToCPU(Results* resultsCPU, Results* resultsGPU, uint32_t nAlignments){
	size_t resSize = nAlignments*sizeof(uint32_t);
	cudaMemcpyKind d2h = cudaMemcpyDeviceToHost;

	resultsCPU->nAlignments = nAlignments;

	resultsCPU->results = (uint32_t*) malloc(resSize);
	cudaMemcpy(resultsCPU->results, resultsGPU->results, resSize, d2h);

	// Check for errors
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();
}


//--Kernel definitions----------------------------------------------------------
__global__ void dtw_align_gpu(DataStruct sequences, MatrixesData matrixes, Results results);


//--Mem free functions----------------------------------------------------------
inline void freeStructsCPU(DataStruct* data, Results* results) {
	// DataStruct free
	free(data->targets);
	free(data->querys);
	free(data->targetRefs);
	free(data->queryRefs);

	// Results free
	free(results->results);
}

inline void freeStructsGPU(DataStruct* data, Results* results) {
	// DataStruct free
	cudaFree(data->targets);
	cudaFree(data->querys);
	cudaFree(data->targetRefs);
	cudaFree(data->queryRefs);

	// Results free
	cudaFree(results->results);

	// Check for errors
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();
}


} // namespace nw

#endif // NW_CUH