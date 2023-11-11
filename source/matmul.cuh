#pragma once

#include <iostream>
#include <stdint.h>
#include <assert.h>
#include <vector>
// #include <algorithm>
#include <numeric>
#include "typedefs.h"

#include "cublas_v2.h"

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
    __shared__ double sh_a[bsize][bsize];
    __shared__ double sh_b[bsize][bsize];

    const uint_fast32_t k_blocks = K / blockDim.x;
    double res = 0.0;

    const bool is_row = row < M;
    const bool is_col = col < N;

    for (uint_fast32_t k = 0; k < k_blocks; ++k) {
        // a_col = k * blockDim.x + threadIdx.y;
        if (is_row)
            sh_a[threadIdx.x][threadIdx.y] = da[row + M * (k * blockDim.x + threadIdx.y)];
        // b_row = k * blockDim.x + threadIdx.x
        if (is_col)
            sh_b[threadIdx.x][threadIdx.y] = db[(k * blockDim.x + threadIdx.x) + K * col];
        __syncthreads();
        for (uint_fast32_t kb = 0; kb < blockDim.x; ++kb) {
            res += sh_a[threadIdx.x][kb] * sh_b[kb][threadIdx.y];
        }
        __syncthreads();
    }
    const uint_fast32_t kr = K % bsize;
    if (kr != 0) {
        if (is_row && threadIdx.y < kr)
            sh_a[threadIdx.x][threadIdx.y] = da[row + M * (k_blocks * blockDim.x + threadIdx.y)];

        if (is_col && threadIdx.x < kr)
            sh_b[threadIdx.x][threadIdx.y] = db[(k_blocks * blockDim.x + threadIdx.x) + K * col];
        __syncthreads();
        for (uint_fast32_t kb = 0; kb < kr; ++kb) {
            res += sh_a[threadIdx.x][kb] * sh_b[kb][threadIdx.y];
        }
    }

    if (is_col && is_row)
        dc[row + M * col] = res;
}


/*
    same as d_matmul_shared, but sh_a is trnasposed so each thread works only with coumns
    , eliminatin memory bank conflicts
*/
__global__ void d_matmul_shared_2(
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
    const uint_fast32_t kr = K % bsize;
    if (kr != 0) {
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

/*
    same as d_matmul_shared, but sh_a and sh_b sizes are changed
    so memory banks do not allign with matrix rows or coumns,
    preventint memory bank conflicts
*/
__global__ void d_matmul_shared_3(
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
    __shared__ double sh_a[bsize][bsize + 1];
    __shared__ double sh_b[bsize][bsize + 1];

    const uint_fast32_t k_blocks = K / blockDim.x;
    double res = 0.0;

    const bool is_row = row < M;
    const bool is_col = col < N;

    for (uint_fast32_t k = 0; k < k_blocks; ++k) {
        // a_col = k * blockDim.x + threadIdx.y;
        if (is_row)
            sh_a[threadIdx.x][threadIdx.y] = da[row + M * (k * blockDim.x + threadIdx.y)];
        // b_row = k * blockDim.x + threadIdx.x
        if (is_col)
            sh_b[threadIdx.x][threadIdx.y] = db[(k * blockDim.x + threadIdx.x) + K * col];
        __syncthreads();
        for (uint_fast32_t kb = 0; kb < blockDim.x; ++kb) {
            res += sh_a[threadIdx.x][kb] * sh_b[kb][threadIdx.y];
        }
        __syncthreads();
    }
    const uint_fast32_t kr = K % bsize;
    if (kr != 0) {
        if (is_row && threadIdx.y < kr)
            sh_a[threadIdx.x][threadIdx.y] = da[row + M * (k_blocks * blockDim.x + threadIdx.y)];

        if (is_col && threadIdx.x < kr)
            sh_b[threadIdx.x][threadIdx.y] = db[(k_blocks * blockDim.x + threadIdx.x) + K * col];
        __syncthreads();
        for (uint_fast32_t kb = 0; kb < kr; ++kb) {
            res += sh_a[threadIdx.x][kb] * sh_b[kb][threadIdx.y];
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
    const uint_fast32_t kernel_type, // type of kernel
    uint_fast32_t n_streams // number of cuda streams
)
{
    uint_fast32_t mem_type = 0; // type of memory allocation of ha, hb, hc
    {
        cudaPointerAttributes a_atr, b_atr, c_atr;
        cudaPointerGetAttributes(&a_atr, ha);
        cudaPointerGetAttributes(&b_atr, hb);
        cudaPointerGetAttributes(&c_atr, hc);
        if (a_atr.type != b_atr.type || a_atr.type != b_atr.type)
            throw std::logic_error("ERROR: not matching memory types are not supported");

        switch (a_atr.type)
        {
        case cudaMemoryTypeHost:
            mem_type = mt_pinned;
            break;
        case cudaMemoryTypeUnregistered:
            mem_type = mt_simple;
            break;
        case cudaMemoryTypeManaged:
            mem_type = mt_unified;
            break;
        default:
            throw std::logic_error("ERROR: memory type is not supported");
            break;
        }
    }
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
    n_streams = stream_cols.size() - 1; // if sizes changed so not all streams will be necessery 
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


    cublasHandle_t handle[n_streams];
    if (kernel_type == cmm_cublas) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            cublasCreate(&handle[st]);
            cublasSetStream(handle[st], stream[st]);
        }
    }

    // pass whole matrix a 
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

    const double alpha = 1;
    const double beta = 0.;
    switch (kernel_type)
    {
    case cmm_simple:
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            dim3 TPB(BLOCKSIZE, BLOCKSIZE);
            dim3 BPG((M + TPB.x - 1) / TPB.x, (stream_cols[st + 1] - stream_cols[st] + TPB.y - 1) / TPB.y);
            d_matmul_1 << <BPG, TPB, 0, stream[st] >> > (
                M,
                stream_cols[st + 1] - stream_cols[st],
                K,
                da,
                db + stream_cols[st] * K,
                dc + stream_cols[st] * M);
        }
        break;
    case cmm_shared:
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            dim3 TPB(BLOCKSIZE, BLOCKSIZE);
            dim3 BPG((M + TPB.x - 1) / TPB.x, (stream_cols[st + 1] - stream_cols[st] + TPB.y - 1) / TPB.y);
            d_matmul_shared << <BPG, TPB, 2 * BLOCKSIZE * BLOCKSIZE * sizeof(double), stream[st] >> > (
                M,
                stream_cols[st + 1] - stream_cols[st],
                K,
                da,
                db + stream_cols[st] * K,
                dc + stream_cols[st] * M);
        }
        break;
    case cmm_shared_2:
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            dim3 TPB(BLOCKSIZE, BLOCKSIZE);
            dim3 BPG((M + TPB.x - 1) / TPB.x, (stream_cols[st + 1] - stream_cols[st] + TPB.y - 1) / TPB.y);
            d_matmul_shared_2 << <BPG, TPB, 2 * BLOCKSIZE * BLOCKSIZE * sizeof(double), stream[st] >> > (
                M,
                stream_cols[st + 1] - stream_cols[st],
                K,
                da,
                db + stream_cols[st] * K,
                dc + stream_cols[st] * M);
        }
        break;
    case cmm_shared_3:
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            dim3 TPB(BLOCKSIZE, BLOCKSIZE);
            dim3 BPG((M + TPB.x - 1) / TPB.x, (stream_cols[st + 1] - stream_cols[st] + TPB.y - 1) / TPB.y);
            d_matmul_shared_3 << <BPG, TPB, 2 * BLOCKSIZE * (BLOCKSIZE + 1) * sizeof(double), stream[st] >> > (
                M,
                stream_cols[st + 1] - stream_cols[st],
                K,
                da,
                db + stream_cols[st] * K,
                dc + stream_cols[st] * M);
        }
        break;
    case cmm_cublas:
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            uint_fast32_t cur_N = stream_cols[st + 1] - stream_cols[st];
            cublasStatus_t stat = cublasDgemm(
                handle[st],
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                M,
                cur_N,
                K,
                &alpha,
                da,
                M,
                db + stream_cols[st] * K,
                K,
                &beta,
                dc + stream_cols[st] * M,
                M);
            assert(!stat);
        }
        break;
    default:
        assert(0);
        break;
    }

    cuda_check();

    if (mem_type == mt_simple || mem_type == mt_pinned) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            cudaMemcpyAsync(
                hc + stream_cols[st] * M,
                dc + stream_cols[st] * M,
                (stream_cols[st + 1] - stream_cols[st]) * M * sizeof(double),
                cudaMemcpyDeviceToHost,
                stream[st]);
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


    if (kernel_type == cmm_cublas) {
        for (uint_fast32_t st = 0; st < n_streams; ++st) {
            cublasDestroy(handle[st]);
        }
    }
    cuda_check();

    cudaDeviceSynchronize();
    for (uint_fast32_t i = 0; i < n_streams; ++i) {
        cudaStreamDestroy(stream[i]);
    }

    cuda_check();
}