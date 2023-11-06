#include "../source/matmul.cuh"
#include "../source/utils.h"


float* benchmark_all(
    const uint32_t n_runs,
    const uint32_t N
) {
    constexpr uint32_t n_funcs = 3;
    const uint32_t N2 = N * N;
    const uint32_t bytes = N * N * sizeof(double);

    double* a, * b, * c, * ha, * hb, * hc, * ma, * mb, * mc;
    // pageable memory
    a = (double*)malloc(bytes);
    b = (double*)malloc(bytes);
    c = (double*)malloc(bytes * n_funcs);
    // pinned memory
    cudaMallocHost(&ha, bytes);
    cudaMallocHost(&hb, bytes);
    cudaMallocHost(&hc, bytes);
    // managed memory
    cudaMallocManaged(&ma, bytes);
    cudaMallocManaged(&mb, bytes);
    cudaMallocManaged(&mc, bytes);

    uint32_t i; // utility function counter

    // time measuring
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float* tot_time;
    tot_time = (float*)malloc(n_funcs * sizeof(float));
    memset(tot_time, 0, n_funcs * sizeof(float));
    float cur_time;

    // time measuring wrapper
    const auto& time_wrap = [&](const auto& f) {
        cudaEventRecord(start, 0);
        cudaEventRecord(stop, 0);

        const double* hc = f();
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cur_time, start, stop);
        tot_time[i] += cur_time;

        if (double* c_cur = c + i * N2;hc != c_cur) // copy result to result matrix array
            memcpy(c_cur, hc, bytes);
        ++i;
        };

    for (uint32_t run = 0; run < n_runs; ++run) {
        i = 0;
        random_matrix(a, N2);
        random_matrix(b, N2);

        memcpy(ha, a, bytes);
        memcpy(hb, b, bytes);
        memcpy(ma, a, bytes);
        memcpy(mb, b, bytes);

        time_wrap(
            [&]() {
                matmul_1_cuda<mt_simple, cmm_simple>(N, N, N, a, b, c + i * N2);
                return c + i * N2;
            });
        time_wrap(
            [&]() {
                matmul_1_cuda<mt_pinned, cmm_simple>(N, N, N, ha, hb, hc);
                return hc;
            });
        time_wrap(
            [&]() {
                matmul_1_cuda<mt_unified, cmm_simple>(N, N, N, ma, mb, mc);
                return mc;
            });


        // check for differences in results
        for (uint_fast8_t j = 0; j < n_funcs - 1; ++j) {
            double dif = mat_diff(&c[j * N2], &c[(j + 1) * N2], N, N);
            if (dif > TOL) {
                printf("Matrixes are not equal for %d and %d, difference: %lf\n", j, j + 1, dif);
                fflush(stdout);
                assert(0);
            }
        }

    }

    for (uint_fast8_t j = 0; j < n_funcs; ++j)
        tot_time[j] /= n_runs;

    // free all arrays
    free(a);
    free(b);
    free(c);
    cudaFree(ma);
    cudaFree(mb);
    cudaFree(mc);
    cudaFreeHost(ha);
    cudaFreeHost(hb);
    cudaFreeHost(hc);

    return tot_time;
}

int main() {

}