#pragma once

#include <random>
#include <malloc.h>
#include <memory.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <cstring>
#include <vector>
#include <string>

#include "utils.h"
#include "typedefs.h"
#include "matmul.cuh"

template <int mem_type> // type from memory_types to use
void cu_benchmark_func(
    blas_dgemm_t* f_arr,
    const uint_fast8_t n_funcs,
    const uint_fast32_t N,
    const uint_fast32_t n_runs,
    const unsigned int seed,
    double* t_sec,
    double* flops)
{
    static_assert(mem_type >= 0 || mem_type < mt_last);
    srand(seed);
    const uint_fast32_t N2 = N * N;
    uint_least64_t mat_bytes = N2 * sizeof(double);
    double* a = (double*)malloc(mat_bytes);
    double* b = (double*)malloc(mat_bytes);
    double* c_arr = (double*)malloc(n_funcs * mat_bytes); // array of result matrixes
    long long t_nano[n_funcs];
    memset(t_nano, 0, n_funcs * sizeof(long long));
    memset(flops, 0, n_funcs * sizeof(double));
    for (uint_fast32_t i = 0; i < n_runs; ++i) {
        for (uint_fast32_t i = 0; i < N2; ++i)
            a[i] = (double)rand() / RAND_MAX;
        for (uint_fast32_t i = 0; i < N2; ++i)
            b[i] = (double)rand() / RAND_MAX;

        for (uint_fast8_t j = 0; j < n_funcs; ++j) { // run all functions on random matrix// use different cuda memory approaches
            double* ha;
            double* hb;
            double* hc;
            if constexpr (mem_type == mt_simple) { // C arrays
                ha = a;
                hb = b;
                hc = &c_arr[j * N2];
            }
            else if constexpr (mem_type == mt_pinned) {
                cudaMallocHost((void**)&ha, mat_bytes);
                cudaMallocHost((void**)&hb, mat_bytes);
                cudaMallocHost((void**)&hc, mat_bytes);

                std::memcpy(ha, a, mat_bytes);
                std::memcpy(hb, b, mat_bytes);
            }
            else if constexpr (mem_type == mt_unified) {
                cudaMallocManaged(&ha, mat_bytes);
                cudaMallocManaged(&hb, mat_bytes);
                cudaMallocManaged(&hc, mat_bytes);

                std::memcpy(ha, a, mat_bytes);
                std::memcpy(hb, b, mat_bytes);
            }


            const auto& start = std::chrono::high_resolution_clock::now();
            (*(f_arr[j]))(N, N, N, ha, hb, hc);
            const auto& finish = std::chrono::high_resolution_clock::now();
            t_nano[j] += (std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)).count();


            // free and transfer different cuda memory approaches
            if constexpr (mem_type == mt_pinned) {
                std::memcpy(&c_arr[j * N2], hc, mat_bytes);
                cudaFreeHost(ha);
                cudaFreeHost(hb);
                cudaFreeHost(hc);
            }
            else if constexpr (mem_type == mt_unified) {
                std::memcpy(&c_arr[j * N2], hc, mat_bytes);
                cudaFree(ha);
                cudaFree(hb);
                cudaFree(hc);
            }
        }

        for (uint_fast8_t j = 0; j < n_funcs - 1; ++j) {
            double dif = mat_diff(&c_arr[j * N2], &c_arr[(j + 1) * N2], N, N);
            if (dif > TOL) {
                printf("Matrixes are not equal for %d and %d, difference: %lf\n", j, j + 1, dif);
                fflush(stdout);
                assert(0);
            }
        }
    }

    for (uint_fast8_t j = 0; j < n_funcs; ++j) {
        t_sec[j] = (double)t_nano[j] / (n_runs * 1000000000L);
        flops[j] = 1e-9 * get_flop_count(N, N, N) / (t_sec[j]);
    }
    free(a);
    free(b);
    free(c_arr);
}



float* benchmark_all(
    const uint32_t n_runs,
    const uint32_t N
) {
    constexpr uint32_t n_funcs = mt_last * cmm_last;
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
        cudaEventRecord(start);

        const double* hc = f();
        cudaEventRecord(stop);
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

        for (uint_fast8_t cmm = 0; cmm < cmm_last; ++cmm) {
            time_wrap(
                [&]() {
                    matmul_cuda(N, N, N, a, b, c + i * N2, mt_simple, cmm, 1);
                    return c + i * N2;
                });
            time_wrap(
                [&]() {
                    matmul_cuda(N, N, N, ha, hb, hc, mt_pinned, cmm, 1);
                    return hc;
                });
            time_wrap(
                [&]() {
                    matmul_cuda(N, N, N, ma, mb, mc, mt_unified, cmm, 1);
                    return mc;
                });
        }

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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tot_time;
}





/*
    EXAMPLE

void benchmark_all(
    unsigned int seed,
    uint_fast32_t n_runs,
    uint_fast32_t N)
{
    const uint8_t n_functions = 4;
    double t_sec[n_functions];
    double flops[n_functions];
    blas_dgemm_t functions[] = { cblas_wrapper, blas_dgemm_parallel, blas_dgemm_parallel_2, blas_dgemm_simple };

    benchmark_func(functions, n_functions, N, n_runs, seed, t_sec, flops);
    for (uint_fast8_t i = 0; i < n_functions; ++i) {
        printf("%f %15f\n", t_sec[i], flops[i]);
    }
}

*/