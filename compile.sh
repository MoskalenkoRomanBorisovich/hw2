#!/bin/bash
mkdir -p build

module load module load nvidia_sdk/nvhpc/23.5

nvcc -O3 main.cu -lcublas -o build/main_cu

nvc++ -mp=gpu -O3 main_omp.c -o build/main_omp
# nvc -O3 main.c -o build/main_omp
