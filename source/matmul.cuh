#pragma once

#include <iostream>
#include <stdint.h>
#include <assert.h>
#include <vector>
// #include <algorithm>
#include <numeric>
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

// returns first col of matrix a  for each stream
// std::vector<uint_fast32_t> get_stream_cols(uint_fast32_t N, uint_fast32_t n_streams) {
//     const uint_fast32_t size = N / n_streams;
//     if (size == 0)
//         n_streams = N;
//     std::vector<uint_fast32_t> res(n_streams + 1, size);
//     std::for_each(res.begin(), res.begin() + N % n_streams, [](uint_fast32_t& a) { ++a;});
//     std::exclusive_scan(res.begin(), res.end(), res.begin(), 0);
//     return res;
// }

// same as get_stream_cols, but number of cols is mostly devisible by BLOCKSIZE
std::vector<uint_fast32_t> get_stream_cols_div_by_block(uint_fast32_t N, uint_fast32_t n_streams) {
    uint_fast32_t size = N / n_streams;
    {
        const uint_fast32_t re = size % BLOCKSIZE;
        if (re != 0)
            size += BLOCKSIZE - re;
    }
    const uint_fast32_t used_streams = (N + size - 1) / size;
    std::vector<uint_fast32_t> res(used_streams + 1, size);
    {
        const uint_fast32_t re = N % size;
        if (re != 0) {
            res[used_streams - 1] = re;
        }
    }
    uint_fast32_t sum = 0;
    for (uint_fast32_t& r : res) {
        std::swap(r, sum);
        sum += r;
    }
    return res;
}

}


/*
    simple matrix multiplication kernel
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

/*
    matrix multiplication kernel with shared memory
*/
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


    constexpr uint_fast16_t bsize = BLOCKSIZE;
    __shared__ double sh_a[bsize][bsize]; // transposed
    __shared__ double sh_b[bsize][bsize];

    const uint_fast32_t k_blocks = K / blockDim.x;
    double res = 0.0;

    const bool is_row = row < M;
    const bool is_col = col < N;

    for (uint_fast32_t k = 0; k < k_blocks; ++k) {
        // a_col = k * blockDim.x + threadIdx.y;
        if (is_row)
            sh_a[threadIdx.y][threadIdx.x] = da[row + M * (k * blockDim.x + threadIdx.y)];
        // b_row = k * blockDim.x + threadIdx.x
        if (is_col)
            sh_b[threadIdx.x][threadIdx.y] = db[(k * blockDim.x + threadIdx.x) + K * col];
        __syncthreads();
        for (uint_fast32_t kb = 0; kb < blockDim.x; ++kb) {
            res += sh_a[kb][threadIdx.x] * sh_b[kb][threadIdx.y];
        }
        __syncthreads();
    }

    if (const uint_fast32_t kr = K % bsize; kr != 0) {
        if (is_row && threadIdx.y < kr)
            sh_a[threadIdx.y][threadIdx.x] = da[row + M * (k_blocks * blockDim.x + threadIdx.y)];

        if (is_col && threadIdx.x < kr)
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
    const uint_fast32_t mem_type, // type of memory allocation of ha, hb, hc
    const uint_fast32_t kernel_type, // type of kernel
    uint_fast32_t n_streams // number of cuda streams
)
{
    assert(mem_type < mt_last);
    assert(kernel_type < cmm_last);

    const auto& cuda_check = [&mem_type, &kernel_type]() {
        cudaError_t er = cudaPeekAtLastError();
        if (er != cudaSuccess) {
            std::cout << "cudaError: " << er << "\n";
            std::cout << kernel_type << " " << mem_type << std::endl;
            fflush(stdout);
            assert(0);
        }
        };

    double* da;
    double* db;
    double* dc;

    const uint_fast32_t a_size = M * K;
    const uint_fast32_t b_size = K * N;
    const uint_fast32_t c_size = M * N;

    const std::vector<uint_fast32_t>& stream_cols = get_stream_cols_div_by_block(M, n_streams);
    n_streams = stream_cols.size() - 1; // if sies changed so not all streams are necessery 
    cudaStream_t stream[n_streams];
    for (uint_fast32_t st = 0; st < n_streams; ++st) {
        cudaStreamCreate(&stream[st]);
    }

    cuda_check();

    // for (const auto i : stream_cols) {
    //     std::cout << i << " ";
    // }
    // std::cout << "\n";

    int device;
    cudaGetDevice(&device);

    cuda_check();


    // pass matrix a whole
    if (mem_type == mt_simple || mem_type == mt_pinned) {
        cudaMalloc((void**)&da, a_size * sizeof(double));
        cudaMalloc((void**)&db, b_size * sizeof(double));
        cudaMalloc((void**)&dc, c_size * sizeof(double));
        cudaMemcpy(da, ha, a_size * sizeof(double), cudaMemcpyHostToDevice);
    }
    else if (mem_type == mt_unified) {
        da = ha;
        db = hb;
        dc = hc;
        cudaMemPrefetchAsync(da, a_size * sizeof(double), device);
    }

    cuda_check();

    cudaDeviceSynchronize();
    // pass matrix b by streams
    if (mem_type == mt_simple || mem_type == mt_pinned) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            cudaMemcpyAsync(
                db + stream_cols[st] * K,
                hb + stream_cols[st] * K,
                (stream_cols[st + 1] - stream_cols[st]) * K * sizeof(double),
                cudaMemcpyHostToDevice,
                stream[st]);
        }
    }
    else if (mem_type == mt_unified) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            cudaMemPrefetchAsync(
                db + stream_cols[st] * K,
                (stream_cols[st + 1] - stream_cols[st]) * K * sizeof(double),
                device,
                stream[st]);
        }
    }

    cuda_check();

    if (kernel_type == cmm_simple) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            dim3 TPB(BLOCKSIZE, BLOCKSIZE);
            dim3 BPG((M + TPB.x - 1) / TPB.x, (stream_cols[st + 1] - stream_cols[st] + TPB.y - 1) / TPB.y);
            d_matmul_1 << <BPG, TPB >> > (
                M,
                stream_cols[st + 1] - stream_cols[st],
                K,
                da,
                db + stream_cols[st] * K,
                dc + stream_cols[st] * M);
        }
    }
    else if (kernel_type == cmm_shared) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            dim3 TPB(BLOCKSIZE, BLOCKSIZE);
            dim3 BPG((M + TPB.x - 1) / TPB.x, (stream_cols[st + 1] - stream_cols[st] + TPB.y - 1) / TPB.y);
            d_matmul_shared << <BPG, TPB >> > (
                M,
                stream_cols[st + 1] - stream_cols[st],
                K,
                da,
                db + stream_cols[st] * K,
                dc + stream_cols[st] * M);
        }
    }

    cuda_check();

    if (mem_type == mt_simple || mem_type == mt_pinned) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            cudaMemcpyAsync(
                hc + stream_cols[st] * M,
                dc + stream_cols[st] * M,
                (stream_cols[st + 1] - stream_cols[st]) * M * sizeof(double), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
    }
    else if (mem_type == mt_unified) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            cudaMemPrefetchAsync(
                dc + stream_cols[st] * M,
                (stream_cols[st + 1] - stream_cols[st]) * M * sizeof(double),
                cudaCpuDeviceId,
                stream[st]);
        }
    }

    cuda_check();

    cudaDeviceSynchronize();
    for (uint_fast32_t i = 0; i < n_streams; ++i) {
        cudaStreamDestroy(stream[i]);
    }

    cuda_check();
}