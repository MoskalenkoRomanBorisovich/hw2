#!/bin/bash
mkdir -p build

module load module load nvidia_sdk/nvhpc/23.5

nvcc -O3 main.cu -o build/main_cu
# nvc -O3 main.c -o build/main_omp
