#include <omp.h>
#include <stdint.h>
#include "source/utils.h"
#include <assert.h>

void gpu_dgemm_omp(
    const uint_fast32_t M,
    const uint_fast32_t N,
    const uint_fast32_t K,
    const double* a,
    const double* b,
    double* c)
{

#pragma omp target teams distribute parallel for map(to:a[0:M*K]) map(to:b[0:K * N]) map(from:c[0:M * N])
    for (uint_fast32_t col = 0; col < N; ++col) {
        for (uint_fast32_t row = 0; row < M; row++) {
            c[col * M + row] = 0.0;
            for (uint_fast32_t k = 0; k < K; k++) {
                c[col * M + row] += a[row + M * k] * b[k + col * K];
            }
        }
    }
}

void test_omp() {
    const int N = 3;
    double a[9] = { 1,2,3,4,5,6,7,8,9 };
    double b[9] = { 9,8,7,6,5,4,3,2,1 };
    double c[9];
    double c_correct[9] = { 9, 114, 138, 54, 69, 84, 18, 24, 30 };
    gpu_dgemm_omp(N, N, N, a, b, c);


    double dif mat_diff(c, c_correct);

    assert(dif < TOL);
}



int main(int argc, char* argv[])
{
    uint32_t N = 1000;
    uint32_t n_runs = 10;
    uint32_t seed = 123;

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

    srand(seed);

    uint32_t bytes = N * N * sizeof(double);
    double* hA, * hB, * hC;
    hA = (double*)malloc(bytes);
    hB = (double*)malloc(bytes);
    hC = (double*)malloc(bytes);

    double time = 0.0, start, stop;
    for (uint32_t run = 0; run < n_runs; ++run) {
        random_matrix(hA, N * N);
        random_matrix(hB, N * N);

        start = omp_get_wtime();
        gpu_dgemm_omp(N, N, N, hA, hB, hC);
        stop = omp_get_wtime();
        time += stop - start;
    }
    time /= n_runs;
    printf("%5s | %10s | %10s", "N", "Time (ms)", "GFLOPS");
    printf("\n");
    printf("%5d | %10f | %10.2f", N, time * 1000, 2.0 * N * N * N / time / 1e9);
    printf("\n");
    free(hA);
    free(hB);
    free(hC);
    return 0;
}