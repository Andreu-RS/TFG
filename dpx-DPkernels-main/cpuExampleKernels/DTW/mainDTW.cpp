//--Includes--------------------------------------------------------------------
#include "DTW.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <chrono>

//--Main------------------------------------------------------------------------
int main(int argc, char **argv) {
    std::cout << "Execution starting..." << std::endl;

	//--Parse arguments---------------------------------------------------------
	std::cout << "Parsing parameters..." << std::endl;

    std::string filename;

    if (argc != 2) {
        std::cout << "ERROR: Wrong parameters" << std::endl;
        std::cout << "Usage: ./exe DTW.seq" << std::endl;
    } 
    else {
        filename = argv[1];
    }

    //--Files managment---------------------------------------------------------
	std::cout << "Reading files..." << std::endl;

    std::vector<std::vector<int>> targetSeqsV;
    std::vector<std::vector<int>> querySeqsV;
    std::vector<uint16_t> targetLensV;
    std::vector<uint16_t> queryLensV;
    uint16_t maxTargetLen = 0;
    uint16_t maxQueryLen = 0;

    std::ifstream file;
    file.open(filename, std::fstream::in);
    if (file.is_open()) {
        /*	NOTE:
		    - In our case we use .seq files so we use one file with all the 
              data,line has starts with < or > to distinguish the pair.
        */

        //--File reading--
        std::string line;
        while(getline(file, line)) { // Read line

            //--Auxiliars--
            std::stringstream lineStream;
            std::vector<int> signal;
            int sample;

            switch (line[0]) { // Look at first character
                case '>':             
                    line.erase(0, 1); // Erase first character from line
                    lineStream.str(line); // String to stream
                    while(lineStream >> sample) { // Read stream
                        signal.push_back(sample); 
                    }
                    targetSeqsV.push_back(signal);
                    targetLensV.push_back(signal.size());
                    maxTargetLen = MAX(maxTargetLen, signal.size());
                    break;
                case '<':
                    line.erase(0, 1);
                    lineStream.str(line);
                    while(lineStream >> sample) {
                        signal.push_back(sample);
                    }
                    querySeqsV.push_back(signal);
                    queryLensV.push_back(signal.size());
                    maxQueryLen = MAX(maxQueryLen, signal.size());
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

    /*  NOTE:
        - For benchmarking purposes we want a certain amount of alignments that
          keep the work (cells computed) constant.
        - In our case we will only have 1 target and 1 query and align them
          several times. 
    */
    int nAlignments = 12000000000 / (targetLensV[0] * queryLensV[0]);


    //--DTW alignment------------------------------------------------------------
    std::cout << "Aligning signals..." << std::endl;

    int** matrix;
    int score;

    /*  NOTE:
        - We need to add 1 extra row and column for the initialization condition
          that needs to be considered when accessing the matrix.
    */
    matrix = new int* [maxQueryLen+1];
        for(int rowIdx = 0; rowIdx <= maxQueryLen; rowIdx++){
            matrix[rowIdx] = new int[maxTargetLen+1];
        }


    // Time control (Accumulator)
    float kernelTimeMicro = 0;

    for(int alIdx = 0; alIdx < nAlignments; alIdx++) {
        /*  NOTE:
            - As mentioned we have 1 single pair that we align multiple times 
              for benchmark purpouses. Otherwise we should use querySeqsV[alIdx]
              and the same for the other parameters.
        */

        // Time control (start)
        std::chrono::steady_clock::time_point begin = 
            std::chrono::steady_clock::now();

        // Kernel execution
        dtw_align(
            querySeqsV[0], 
            queryLensV[0], 
            targetSeqsV[0], 
            targetLensV[0], 
            matrix,
            &score);
        
        // Time control (end and accumulate)
        std::chrono::steady_clock::time_point end 
            = std::chrono::steady_clock::now();
        kernelTimeMicro += 
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        /*
        dtw_out(
            alIdx, 
            querySeqsV[0], 
            queryLensV[0], 
            targetSeqsV[0], 
            targetLensV[0], 
            matrix,
            &score);
        */
    }

    //--Bench results-----------------------------------------------------------
    std::cout << "Gathering results..." << std::endl;

    // Compute values
    const double seconds = kernelTimeMicro / 1000000;
    const double gCells = 
        ((double) targetLensV[0] * queryLensV[0] * nAlignments) / 1000000000L;
    
    // Performance info
    std::cout << "Performance results:"
        << "\n\tTotal number of alignments: " << nAlignments
        << "\n\tTotal number of cells (GCells): " << gCells
        << "\n\tKernel execution time (s):  " << seconds
        << "\n\tGCells/s: " << gCells / seconds
        << std::endl;


    return 0;
}
