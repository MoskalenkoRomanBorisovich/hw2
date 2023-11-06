#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


#define TOL 1e-6
/*
    make random double matrix M x N
    without memory allocation
*/
void random_matrix(double* a, uint32_t size) {
    for (uint_fast32_t i = 0; i < size; ++i) {
        if (rand() % 2 == 0)
            a[i] = (double)rand() / RAND_MAX;
        else
            a[i] = -(double)rand() / RAND_MAX;
    }
}

void random_mod10_mat(double* a, uint32_t size) {
    for (uint_fast32_t i = 0; i < size; ++i) {
        if (rand() % 2 == 0)
            a[i] = rand() % 10;
        else
            a[i] = -rand() % 10;
    }
}


/*
    prints column-major matrix
*/
void print_matrix(double* a, uint_fast64_t n, uint_fast64_t m)
{
    for (uint_fast64_t i = 0; i < n; i++) {
        for (uint_fast64_t j = 0; j < m; j++) {
            printf("%f ", a[j * n + i]);
        }
        printf("\n");
    }
}


/*
number of operations for matrix multiplication
*/
uint_fast64_t get_flop_count(uint_fast64_t N, uint_fast64_t M, uint_fast64_t K) {
    return 2 * N * M * K;
}


/*
squared Frobenius norm of matrix difference
*/
double mat_diff(const double* a, const double* b, uint_fast64_t N, uint_fast64_t M) {
    double res = 0.0;
    double d;
    for (uint_fast64_t i = 0, i_end = N * M; i < i_end; ++i) {
        d = a[i] - b[i];
        res += d * d;
    }
    return res;
}