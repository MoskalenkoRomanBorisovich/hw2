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
    "Simple",
    "Pinned",
    "Unified"
};

// cuda matrix multiplication kernels
enum cmm_kernels {
    cmm_simple,
    cmm_shared,
    cmm_shared_2,
    cmm_shared_3,
    cmm_last,
};

char kernel_names[cmm_last][15] = {
    "Simple",
    "Shared",
    "Shared_2",
    "Shared_3",
};