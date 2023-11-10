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

void benchmark_all(
    const uint32_t n_runs,
    const uint32_t N
) {
    constexpr uint8_t stream_pow_max = 3;
    constexpr uint32_t n_funcs = mt_last * cmm_last * (stream_pow_max + 1);
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

    uint32_t i = 0; // utility function counter

    // time measuring
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<float> tot_time(n_funcs, 0.0);
    float cur_time;

    // time measuring wrapper
    const auto& time_wrap = [&](const auto& f) {
        cudaEventRecord(start);

        const double* hc = f();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cur_time, start, stop);
        tot_time[i] += cur_time;
        double* c_cur = c + i * N2;
        if (hc != c_cur) // copy result to result matrix array
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
        uint32_t n_streams = 1;
        for (uint32_t stream_pow = 0; stream_pow <= stream_pow_max; ++stream_pow) {
            for (uint_fast8_t cmm = 0; cmm < cmm_last; ++cmm) {
                time_wrap(
                    [&]() {
                        matmul_cuda(N, N, N, a, b, c + i * N2, mt_simple, cmm, n_streams);
                        return c + i * N2;
                    });
                time_wrap(
                    [&]() {
                        matmul_cuda(N, N, N, ha, hb, hc, mt_pinned, cmm, n_streams);
                        return hc;
                    });
                time_wrap(
                    [&]() {
                        matmul_cuda(N, N, N, ma, mb, mc, mt_unified, cmm, n_streams);
                        return mc;
                    });
            }
            n_streams *= 2;
        }
        // check for differences in results
        for (uint_fast8_t j = 0; j < n_funcs - 1; ++j) {
            double dif = mat_diff(&c[j * N2], &c[(j + 1) * N2], N, N);
            if (dif > TOL) {
                printf("\n\nOn run %i matrixes are not equal for %d and %d, difference: %lf\n", run, j, j + 1, dif);
                fflush(stdout);

                constexpr int s = 5;
                std::cout << "\nmatrix A:\n";
                print_matrix(a, N, N, s, s);
                std::cout << "\nmatrix B:\n";
                print_matrix(b, N, N, s, s);
                std::cout << "\nfirst matrix:\n";
                print_matrix(c + j * N2, N, N, s, s);
                std::cout << "\nsecond matrix:\n";
                print_matrix(c + (j + 1) * N2, N, N, s, s);
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

    double gflop_ms = get_flop_count(N, N, N) * 1e-6;
    { // pretty print also in csv format
        printf("%5s | %7s | %10s | %10s | %15s | %15s\n", "N", "streams", "kernel", "memory", "Time (ms)", "GFLOPS");
        printf("\n");
        uint32_t n_streams = 1;
        uint32_t i = 0;
        for (int stream_pow = 0; stream_pow <= stream_pow_max; ++stream_pow) {
            for (int cmm = 0; cmm < cmm_last; ++cmm) {
                for (int mem = 0; mem < mt_last; ++mem) {
                    printf("%5i | %7i | %10s | %10s | %15f | %15f\n", N, n_streams, kernel_names[cmm], memory_names[mem], tot_time[i], gflop_ms / tot_time[i]);
                    ++i;
                }
                printf("\n");
            }
            n_streams *= 2;
        }
    }
}