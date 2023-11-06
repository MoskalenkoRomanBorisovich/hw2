#pragma once


typedef void (*blas_dgemm_t)(const uint_fast32_t M, const uint_fast32_t N, const uint_fast32_t K, double* a, double* b, double* c);

// type of CUDA memory management 
enum memory_types {
    mt_simple = 0, // just c arrays
    mt_pinned, // using CUDA pinned memory
    mt_unified, // using CUDA unified memory
    mt_last // dummy, always last
};

// string names of corrsponding types form memory_types
char memory_names[mt_last][15] = {
    "mt_simple",
    "mt_pinned",
    "mt_unified"
};

// cuda matrix multiplication kernels
enum cmm_kernels {
    cmm_simple = 0,
    cmm_shared,
    cmm_last
};

char kernel_names[cmm_last][15] = {
    "cmm_simple",
    "cmm_shared",
};