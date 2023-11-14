#!/bin/bash
mkdir -p build

module load nvidia_sdk/nvhpc/23.5

echo compiling cuda
# export NVCC_PREPEND_FLAGS='-ccbin  /usr/bin/gcc'
nvcc -O3 main.cu -lcublas -o build/main_cu
echo complete

echo compiling openmp
nvc++ -mp=gpu -O3 main_omp.cpp -o build/main_omp
echo complete
# nvc -O3 main.c -o build/main_omp
