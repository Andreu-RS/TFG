//--Includes--------------------------------------------------------------------
#include "DTW.cuh"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string.h>

using namespace dtw; // DTW.cuh

//--Main------------------------------------------------------------------------
int main(int argc, char **argv) {
    std::cout << "Execution starting..." << std::endl;

	//--Parse arguments---------------------------------------------------------
	std::cout << "Parsing parameters..." << std::endl;

    std::string filename;

    if (argc != 2) {
        std::cout << "ERROR: Wrong parameters" << std::endl;
        std::cout << "Usage: ./exe NW.seq" << std::endl;
    } 
    else {
        filename = argv[1];
    }


    //--Files managment---------------------------------------------------------
	std::cout << "Reading files..." << std::endl;

    // Struct to hold all the required data (see NW.cuh)
    DataStruct sequencesData; 
    initializeDataStruct(&sequencesData);

    // File reading
    std::ifstream file;
    file.open(filename, std::fstream::in);
    if (file.is_open()) {
        /*	NOTE:
		    - In our case we use .seq files so we use one file with all the 
              data,line starts with < or > to distinguish the pair.
            - Note that we DO NOT want the 1st char in the line.
        */

        // Line reading
        std::string line; 
        while(getline(file, line)) { // Read line

            // Auxiliars
            std::stringstream lineStream;
            std::vector<uint32_t> signal;
            uint32_t sample;

            switch (line[0]) { // Look at first character
                case '>':
                    line.erase(0, 1); // Erase first character from line
                    lineStream.str(line); // String to stream
                    while(lineStream >> sample) { // Read stream
                        signal.push_back(sample);
                    }
                    storeTarget(&(signal[0]), signal.size(), &sequencesData);
                    break;
                case '<':
                    line.erase(0, 1); // Erase first character from line
                    lineStream.str(line); // String to stream
                    while(lineStream >> sample) { // Read stream
                        signal.push_back(sample); 
                    }
                    storeQuery(&(signal[0]), signal.size(), &sequencesData);
                    break;
                default:
                    std::cout << "ERROR: problem reading files" << std::endl;
                    return -1;
            }
        }
        file.close();
    } 
    else {
        std::cout << "ERROR: File " << filename << " not found." << std::endl;
        return -1;
    }

    /*
    //TEST CODE START
        std::cout << "Total number of targets/querys: " << sequencesData.nTargets << " / "
            << sequencesData.nQuerys
            << "\nTotal elements in targets/querys: "
            << sequencesData.targetRefs[sequencesData.nTargets] << " / "
            << sequencesData.queryRefs[sequencesData.nQuerys]<< std::endl;
        
        for(int i = 0; i < sequencesData.nTargets; i++) {
            std::cout << "\nTarget "<< i+1 << ": (reference " << sequencesData.targetRefs[i] << ")\n";
            for (int t = sequencesData.targetRefs[i]; t < sequencesData.targetRefs[i+1]; t++) {
                std::cout << sequencesData.targets[t] << " ";
            }

            std::cout << "\nQuery "<< i+1 << ": (reference " << sequencesData.queryRefs[i] << ")\n";
            for (int q = sequencesData.queryRefs[i]; q < sequencesData.queryRefs[i+1]; q++) {
                std::cout << sequencesData.querys[q] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    //TEST CODE END 
    */


    //--Parameter definitions---------------------------------------------------
    std::cout << "Defining parameters..." << std::endl;

    // General parameters
    /*  NOTE:
        - For benchmarking purposes we want a certain amount of alignments that
          keep the work (cells computed) constant.
        - In our case we will only have 1 target and 1 query and align them
          several times. 
    */
    int nAlignments = 12000000000 /
        (sequencesData.targetRefs[1] * sequencesData.queryRefs[1]);

    // Kernel launching parameters
    int nThreadsPerBlock = 32; //TODO: parameterize
    int nBlocks = nAlignments/nThreadsPerBlock;
    if(nAlignments % nThreadsPerBlock) {
        nBlocks++;
    }


    //--Set GPU environment up--------------------------------------------------
    std::cout << "Preparing GPU..." << std::endl;

    /*  NOTE:
        - Pointers in GPU structures are DEVICE pointers, used only in GPU.
        - sequencesDataGPU does NOT need to be initialized, all pointers are 
          initialized during the copy to the GPU.
        - When passing a struct to the kernel a deep copy is made.
    */

    // Allocate and initialize results struct in GPU
    Results resultsGPU;
    initializeResultsGPU(&resultsGPU, nAlignments);

    // Set matrixesData
    MatrixesData matrixesGPU;
    setMatrixesDataGPU(&matrixesGPU, &sequencesData, nAlignments);

    // Copy data required to perform the alignments
    DataStruct sequencesDataGPU;
    copySequencesToGPU(&sequencesData, &sequencesDataGPU);


    //--DTW alignment-----------------------------------------------------------
    std::cout << "Launching kernels...\n\tnBlocks = " << nBlocks 
        << "\tnThreadsPerBlock = " << nThreadsPerBlock << std::endl;

    // Create and start counter
    TIMER_CREATE;
    TIMER_START;

    // Perform aligments in GPU
    dtw_align_gpu<<<nBlocks, nThreadsPerBlock>>>(
        sequencesDataGPU, 
        matrixesGPU,
        resultsGPU);
    
    // Stop counter and get time
    TIMER_STOP; 
    TIME_ELAPSED(kernelTime);

    // Check for error
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();


    //--Recover data from GPU---------------------------------------------------
    std::cout << "Recovering results..." << std::endl;

    // Copy results to CPU
    Results resultsCPU;
    copyResultsToCPU(&resultsCPU, &resultsGPU, nAlignments);


    //--Print results-----------------------------------------------------------
    // Compute values
    const double seconds = kernelTime / 1000;
    const double gCells = ((double) sequencesData.queryRefs[1] 
        * sequencesData.targetRefs[1] * nAlignments) / 1000000000L;
    
    // Performance info
    std::cout << "Performance results:"
        << "\n\tTotal number of alignments: " << nAlignments
        << "\n\tTotal number of cells (GCells): " << gCells
        << "\n\tKernel execution time (s):  " << seconds
        << "\n\tGCells/s: " << gCells / seconds
        << std::endl;

    // Alignments info
    /*
    std::cout << "Alignment results:";
    for(int i = 0; i < nAlignments; i++) {
        std::cout << "\n\tAlIdx: " << i 
            << "\n\tScore: " << resultsCPU.results[i] 
            << std::endl;
    }
    std::cout << std::endl;
    */


    //--Freeing memory----------------------------------------------------------
    std::cout << "Freeing memory..." << std::endl;

    freeStructsCPU(&sequencesData, &resultsCPU);
    freeStructsGPU(&sequencesDataGPU, &resultsGPU);


    return 0;
}
