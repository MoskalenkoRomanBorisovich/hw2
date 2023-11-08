#pragma once

#include <iostream>
#include <stdint.h>
#include <assert.h>

#include "typedefs.h"

/*
  a [M][K]
  b [K][N]
  c [M][N]

  a[i, j] = i + M * j
  b[i, j] = i + K * j
  c[i, j] = i + M * j
*/

/*
    ========================== V1 ==========================
*/


namespace {
#define BLOCKSIZE 32
}
/*
    simple device matrix multiplication
*/
__global__ void d_matmul_1(
    const uint_fast32_t M,
    const uint_fast32_t N,
    const uint_fast32_t K,
    const double* da,
    const double* db,
    double* dc)
{
    const uint_fast32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const uint_fast32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= M || j >= N)
        return;

    dc[i + M * j] = 0;
    for (uint_fast32_t k = 0; k < K; ++k) {
        dc[i + M * j] += da[i + M * k] * db[k + K * j];
    }
}


__global__ void d_matmul_shared(
    const uint_fast32_t M,
    const uint_fast32_t N,
    const uint_fast32_t K,
    const double* da,
    const double* db,
    double* dc)
{
    // assume blockDim.x == blockDim.y
    const uint_fast32_t row = blockDim.x * blockIdx.x + threadIdx.x;
    const uint_fast32_t col = blockDim.y * blockIdx.y + threadIdx.y;


    constexpr uint_fast16_t bsize = 32;
    __shared__ double sh_a[bsize][bsize]; // transposed
    __shared__ double sh_b[bsize][bsize];

    const uint_fast32_t k_blocks = K / blockDim.x;
    double res = 0.0;

    const bool is_row = row < M;
    const bool is_col = col < N;

    for (uint_fast32_t k = 0; k < k_blocks; ++k) {
        // a_col = k * blockDim.x + threadIdx.y;
        sh_a[threadIdx.y][threadIdx.x] = da[row + M * (k * blockDim.x + threadIdx.y)];
        // b_row = k * blockDim.x + threadIdx.x
        sh_b[threadIdx.x][threadIdx.y] = db[(k * blockDim.x + threadIdx.x) + K * col];
        __syncthreads();
        for (uint_fast32_t kb = 0; kb < blockDim.x; ++kb) {
            res += sh_a[kb][threadIdx.x] * sh_b[kb][threadIdx.y];
        }
        __syncthreads();
    }

    if (const uint_fast32_t kr = K % bsize; kr != 0) {
        sh_a[threadIdx.y][threadIdx.x] = da[row + M * (k_blocks * blockDim.x + threadIdx.y)];

        sh_b[threadIdx.x][threadIdx.y] = db[(k_blocks * blockDim.x + threadIdx.x) + K * col];
        __syncthreads();
        for (uint_fast32_t kb = 0; kb < kr; ++kb) {
            res += sh_a[kb][threadIdx.x] * sh_b[kb][threadIdx.y];
        }
    }

    if (is_col && is_row)
        dc[row + M * col] = res;
}



void matmul_cuda(
    const uint_fast32_t M,
    const uint_fast32_t N,
    const uint_fast32_t K,
    double* ha,
    double* hb,
    double* hc,
    const uint_fast8_t mem_type, // type of memory allocation of ha, hb, hc
    const uint_fast8_t kernel_type, // type of kernel
    const uint_fast8_t n_streams // number of cuda streams
)
{
    assert(mem_type < mt_last);
    assert(kernel_type < cmm_last);
    double* da;
    double* db;
    double* dc;

    const uint_fast32_t a_size = M * K;
    const uint_fast32_t b_size = K * N;
    const uint_fast32_t c_size = M * N;

    cudaStream_t stream[n_streams];
    if (mem_type == mt_simple || mem_type == mt_pinned) {
        cudaMalloc((void**)&da, a_size * sizeof(double));
        cudaMalloc((void**)&db, b_size * sizeof(double));
        cudaMalloc((void**)&dc, c_size * sizeof(double));
        cudaMemcpy(da, ha, a_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, b_size * sizeof(double), cudaMemcpyHostToDevice);
    }
    else if (mem_type == mt_unified) {
        int device;
        cudaGetDevice(&device);
        da = ha;
        db = hb;
        dc = hc;
        cudaMemAdvise(da, a_size * sizeof(double), cudaMemAdviseSetReadMostly, device);
        cudaMemAdvise(db, b_size * sizeof(double), cudaMemAdviseSetReadMostly, device);
        cudaMemPrefetchAsync(da, a_size * sizeof(double), device);
        cudaMemPrefetchAsync(db, b_size * sizeof(double), device);
    }

    dim3 TPB(BLOCKSIZE, BLOCKSIZE);
    dim3 BPG((M + TPB.x - 1) / TPB.x, (N + TPB.y - 1) / TPB.y);

    if (kernel_type == cmm_simple) {
        d_matmul_1
            << <BPG, TPB >> >
            (M, N, K, da, db, dc);
    }
    else if (kernel_type == cmm_shared) {
        d_matmul_shared
            << <BPG, TPB >> >
            (M, N, K, da, db, dc);
    }

    if (mem_type == mt_simple || mem_type == mt_pinned) {
        cudaMemcpy(hc, dc, c_size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
    }
    else if (mem_type == mt_unified) {
        cudaDeviceSynchronize();
    }
}


