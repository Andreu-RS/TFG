#!/bin/bash
echo "Compiling..."
g++ cpuExampleKernels/NW/mainNW.cpp cpuExampleKernels/NW/NW.cpp -o cpuNW
g++ cpuExampleKernels/DTW/mainDTW.cpp cpuExampleKernels/DTW/DTW.cpp -o cpuDTW
nvcc -gencode arch=compute_90,code=sm_90 gpuExampleKernels/NW/mainNW.cu gpuExampleKernels/NW/NW.cu -o gpuNW
nvcc -gencode arch=compute_90,code=sm_90 gpuExampleKernels/DTW/mainDTW.cu gpuExampleKernels/DTW/DTW.cu -o gpuDTW
echo "============================"
echo "Running cpuNW (DS 10K)"
./cpuNW DataSinglePairs/singlePairNW_10K.seq
echo "----------------------------"
echo "Running cpuNW (DS 1K)"
./cpuNW DataSinglePairs/singlePairNW_1K.seq
echo "----------------------------"
echo "Running cpuNW (DS 250)"
./cpuNW DataSinglePairs/singlePairNW_250.seq
echo "----------------------------"
echo "Running cpuNW (DS 100)"
./cpuNW DataSinglePairs/singlePairNW_100.seq

echo "============================"
echo "Running cpuDTW (DS 10K)"
./cpuDTW DataSinglePairs/singlePairDTW_10K.seq
echo "----------------------------"
echo "Running cpuDTW (DS 1K)"
./cpuDTW DataSinglePairs/singlePairDTW_1K.seq
echo "----------------------------"
echo "Running cpuDTW (DS 250)"
./cpuDTW DataSinglePairs/singlePairDTW_250.seq
echo "----------------------------"
echo "Running cpuDTW (DS 100)"
./cpuDTW DataSinglePairs/singlePairDTW_100.seq

echo "============================"
echo "Running gpuNW (DS 10K)"
./gpuNW DataSinglePairs/singlePairNW_10K.seq
echo "----------------------------"
echo "Running gpuNW (DS 1K)"
./gpuNW DataSinglePairs/singlePairNW_1K.seq
echo "----------------------------"
echo "Running gpuNW (DS 250)"
./gpuNW DataSinglePairs/singlePairNW_250.seq
echo "----------------------------"
echo "Running gpuNW (DS 100)"
./gpuNW DataSinglePairs/singlePairNW_100.seq

echo "============================"
echo "Running gpuDTW (DS 10K)"
./gpuDTW DataSinglePairs/singlePairDTW_10K.seq
echo "----------------------------"
echo "Running gpuDTW (DS 1K)"
./gpuDTW DataSinglePairs/singlePairDTW_1K.seq
echo "----------------------------"
echo "Running gpuDTW (DS 250)"
./gpuDTW DataSinglePairs/singlePairDTW_250.seq
echo "----------------------------"
echo "Running gpuDTW (DS 100)"
./gpuDTW DataSinglePairs/singlePairDTW_100.seq
