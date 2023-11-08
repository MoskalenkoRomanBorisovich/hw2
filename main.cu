#include "source/matmul.cuh"
#include "source/utils.h"
#include "source/benchmark.cuh"

void benchmark_all(
    unsigned int seed,
    uint_fast32_t n_runs,
    uint_fast32_t N)
{
    constexpr uint8_t n_functions = 6;
    // double t_sec[n_functions];
    // double flops[n_functions];
    char func_names[n_functions][15] = { "simple", "pinned", "managed",  "simple, sh", "pinned, sh", "managed, sh" };
    double gflop_ms = get_flop_count(N, N, N) * 1e-6;
    // cu_benchmark_func<mt_unified>(functions, n_functions, N, n_runs, seed, t_sec, flops);
    float* time_ms = benchmark_all(n_runs, N);
    { // pretty print
        const int l = printf("%15s || %15s || %15s\n", "Function name", "Time (ms)", "GFLOPS") - 1;
        char sep[l + 1];
        sep[l] = 0;
        for (int i = 0; i < l; ++i) sep[i] = '=';
        printf("%s\n", sep);
        for (uint_fast8_t i = 0; i < n_functions; ++i) {
            printf("%15s || %15f || %15f\n", func_names[i], time_ms[i], gflop_ms / time_ms[i]);
        }
        printf("%s\n", sep);
    }

    free(time_ms);
    // cu_benchmark_func<mt_pinned>(functions, n_functions, N, n_runs, seed, t_sec, flops);
    // { // pretty print
    //     const int l = printf("%15s || %15s || %15s\n", "Function name", "Time (sec)", "GFLOPS") - 1;
    //     char sep[l + 1];
    //     sep[l] = 0;
    //     for (int i = 0; i < l; ++i) sep[i] = '=';
    //     printf("%s\n", sep);
    //     for (uint_fast8_t i = 0; i < n_functions; ++i) {
    //         printf("%15s || %15f || %15f\n", func_names[i], t_sec[i], flops[i]);
    //     }
    //     printf("%s\n", sep);
    // }
}

int main(int argc, char** argv) {
    int N = 1000;
    int seed = 123;
    int n_runs = 100;

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

    benchmark_all(seed, n_runs, N);
    return 0;
}