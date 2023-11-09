#include "source/matmul.cuh"
#include "source/utils.h"
#include "source/benchmark.cuh"



int main(int argc, char** argv) {
    int N = 1000;
    int n_runs = 100;
    int seed = 123;

    switch (argc)
    {
    case 4:
        seed = std::atoi(argv[3]);
    case 3:
        n_runs = std::atoi(argv[2]);
    case 2:
        N = std::atoi(argv[1]);

    default:
        break;
    }

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    srand(seed);
    // test_all(n_runs, N);
    benchmark_all(n_runs, N);
    return 0;
}