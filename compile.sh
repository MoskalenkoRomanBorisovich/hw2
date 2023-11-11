#!/bin/bash
mkdir -p build

module load nvidia_sdk/nvhpc/23.5

echo compiling cuda
nvcc -O3 main.cu -lcublas -o build/main_cu
echo complete

echo compiling openmp
nvc++ -mp=gpu -O3 main_omp.c -o build/main_omp
echo complete
# nvc -O3 main.c -o build/main_omp
