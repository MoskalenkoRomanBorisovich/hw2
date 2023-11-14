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
#include <array>

#include "utils.h"
#include "typedefs.h"
#include "matmul.cuh"
#include "cublas_v2.h"

void benchmark_all(
    const uint32_t n_runs,
    const uint32_t N
) {

    const  std::array<int, 3> bsides = { 8, 16, 32 };
    const  uint8_t stream_pow_max = 3;
    const uint32_t n_funcs = mt_last * cmm_last * (stream_pow_max + 1) * bsides.size();
    const uint32_t N2 = N * N;
    const uint32_t bytes = N * N * sizeof(double);

    double* a, * b, * c_cur, * c_prev, * ha, * hb, * hc, * ma, * mb, * mc;
    // pageable memory
    a = (double*)malloc(bytes);
    b = (double*)malloc(bytes);
    c_cur = (double*)malloc(bytes);
    c_prev = (double*)malloc(bytes);
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
        if (hc != c_cur) // copy result to result matrix array
            memcpy(c_cur, hc, bytes);

        if (i == 0) {
            std::swap(c_cur, c_prev);
        }
        else { // check differense i result
            double dif = mat_diff(c_prev, c_cur, N, N);
            if (dif > TOL) {
                printf("\n\nMatrixes are not equal for %d and %d, difference: %lf\n", i, i - 1, dif);
                fflush(stdout);

                const uint32_t s = std::min((uint32_t)5, N); // submatrix size to print
                std::cout << "\nmatrix A:\n";
                print_matrix(a, N, N, s, s);
                std::cout << "\nmatrix B:\n";
                print_matrix(b, N, N, s, s);
                std::cout << "\nfirst matrix:\n";
                print_matrix(c_prev, N, N, s, s);
                std::cout << "\nsecond matrix:\n";
                print_matrix(c_cur, N, N, s, s);
                assert(0);
            }
        }
        ++i;
        };

    // main loop
    for (uint32_t run = 0; run < n_runs; ++run) {
        i = 0;
        random_matrix(a, N2);
        random_matrix(b, N2);

        memcpy(ha, a, bytes);
        memcpy(hb, b, bytes);
        memcpy(ma, a, bytes);
        memcpy(mb, b, bytes);
        uint32_t n_streams = 1;
        // measure all variants of my functions
        for (uint32_t stream_pow = 0; stream_pow <= stream_pow_max; ++stream_pow) {
            for (uint_fast8_t cmm = 0; cmm < cmm_last; ++cmm) {
                for (const int bs : bsides) {
                    time_wrap(
                        [&]() {
                            matmul_cuda(N, N, N, a, b, c_cur, cmm, n_streams, bs);
                            return c_cur;
                        });
                    time_wrap(
                        [&]() {
                            matmul_cuda(N, N, N, ha, hb, hc, cmm, n_streams, bs);
                            return hc;
                        });
                    time_wrap(
                        [&]() {
                            matmul_cuda(N, N, N, ma, mb, mc, cmm, n_streams, bs);
                            return mc;
                        });
                }
            }
            n_streams *= 2;
        }
    }

    // get mean time
    for (uint_fast8_t j = 0; j < n_funcs; ++j)
        tot_time[j] /= n_runs;

    // free all arrays
    free(a);
    free(b);
    free(c_cur);
    free(c_prev);
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
        printf("%5s | %5s | %7s | %10s | %10s | %15s | %15s\n", "N", "bside", "streams", "kernel", "memory", "Time (ms)", "GFLOPS");
        printf("\n");
        uint32_t n_streams = 1;
        uint32_t i = 0;
        for (int stream_pow = 0; stream_pow <= stream_pow_max; ++stream_pow) {
            for (int cmm = 0; cmm < cmm_last; ++cmm) {
                for (const int& bs : bsides) {
                    for (int mem = 0; mem < mt_last; ++mem) {
                        printf("%5i | %5i | %7i | %10s | %10s | %15f | %15f\n", N, bs, n_streams, kernel_names[cmm], memory_names[mem], tot_time[i], gflop_ms / tot_time[i]);
                        ++i;
                    }
                    printf("\n");
                }
            }
            n_streams *= 2;
        }
    }
}