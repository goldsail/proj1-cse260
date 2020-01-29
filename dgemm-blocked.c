/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <stdio.h>  // For: perror
#include <immintrin.h>
#include <avx2intrin.h> // AVX2 Intrinsics
#include <stdlib.h>
#include <string.h>
#include <assert.h>

const char* dgemm_desc = "Multilevel blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 37
#endif
#ifndef L1_BLOCK_SIZE_M
#define L1_BLOCK_SIZE_M 32
#endif
#ifndef L1_BLOCK_SIZE_N
#define L1_BLOCK_SIZE_N 32
#endif
#ifndef L1_BLOCK_SIZE_K
#define L1_BLOCK_SIZE_K 32
#endif
#ifndef L2_BLOCK_SIZE_M
#define L2_BLOCK_SIZE_M 96
#endif
#ifndef L2_BLOCK_SIZE_N
#define L2_BLOCK_SIZE_N 96
#endif
#ifndef L2_BLOCK_SIZE_K
#define L2_BLOCK_SIZE_K 96
#endif
#ifndef L3_BLOCK_SIZE_M
#define L3_BLOCK_SIZE_M 1056
#endif
#ifndef L3_BLOCK_SIZE_N
#define L3_BLOCK_SIZE_N 1056
#endif
#ifndef L3_BLOCK_SIZE_K
#define L3_BLOCK_SIZE_K 1056
#endif

#define VECTOR_SIZE 4
// #define BLOCK_SIZE 719

#define min(a,b) (((a)<(b))?(a):(b))

#ifdef CACHELAYOUT
int dealias(int lda) {
  // heavy cache aliasing if lda near multiples of 128
  // pad some zeros to avoid this
  if (lda % 128 == 127) return lda + 5;
  if (lda % 128 == 0) return lda + 4;
  #ifdef ALIGN
  return (1 + (lda - 1) / 4) * 4; // ceil(lda) to multiple of 4
  #else
  return lda;
  #endif
}

static void copy_layout(int lda, double *A, double *B, double *C, int ldn, double **a, double **b, double **c) {
#ifdef ALIGN
  posix_memalign(a, sizeof(double) * VECTOR_SIZE, sizeof(double) * lda * ldn);
  posix_memalign(b, sizeof(double) * VECTOR_SIZE, sizeof(double) * lda * ldn);
  posix_memalign(c, sizeof(double) * VECTOR_SIZE, sizeof(double) * lda * ldn);
#else
  *a = (double*)malloc(sizeof(double) * lda * ldn);
  *b = (double*)malloc(sizeof(double) * lda * ldn);
  *c = (double*)malloc(sizeof(double) * lda * ldn);
#endif
  for (int i = 0; i < lda; i++) {
    memcpy(*a + i * ldn, A + i * lda, sizeof(double) * lda);
    memset(*a + i * ldn + lda, 0, sizeof(double) * (ldn - lda));
  }
  for (int i = 0; i < lda; i++) {
    memcpy(*b + i * ldn, B + i * lda, sizeof(double) * lda);
    memset(*b + i * ldn + lda, 0, sizeof(double) * (ldn - lda));
  }
  for (int i = 0; i < lda; i++) {
    memcpy(*c + i * ldn, C + i * lda, sizeof(double) * lda);
    memset(*c + i * ldn + lda, 0, sizeof(double) * (ldn - lda));
  }
}

static void copy_back_layout(int ldn, double **a, double **b, double **c, int lda, double *C) {
  for (int i = 0; i < lda; i++) {
    memcpy(C + i * lda, *c + i * ldn, sizeof(double) * lda);
  }
  free(*a);
  free(*b);
  free(*c);
  *a = *b = *c = NULL;
}
#endif

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each row i of A */
  for (int i = 0; i < M; ++i) {
    /* For each column j of B */
    for (int j = 0; j < N; ++j) {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      for (int k = 0; k < K; ++k)
      #ifdef TRANSPOSE
      	cij += A[i*lda+k] * B[j*lda+k];
      #else
      	cij += A[i*lda+k] * B[k*lda+j];
      #endif
      C[i*lda+j] = cij;
    }
  }
}
#ifdef ALIGN
#define My_mm256_loadu_pd(C) _mm256_load_pd(C)
#else
#define My_mm256_loadu_pd(C) _mm256_loadu_pd(C)
#endif

#ifdef ALIGN
#define My_mm256_storeu_pd(C, R) _mm256_store_pd(C, R)
#else
#define My_mm256_storeu_pd(C, R) _mm256_storeu_pd(C, R)
#endif

static void do_block_8x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C + lda*0);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda*1);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + lda*2);
  register __m256d c30_c31_c32_c33 = My_mm256_loadu_pd(C + lda*3);
  register __m256d c40_c41_c42_c43 = My_mm256_loadu_pd(C + lda*4);
  register __m256d c50_c51_c52_c53 = My_mm256_loadu_pd(C + lda*5);
  register __m256d c60_c61_c62_c63 = My_mm256_loadu_pd(C + lda*6);
  register __m256d c70_c71_c72_c73 = My_mm256_loadu_pd(C + lda*7);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A + kk + 4*lda);
    register __m256d a5x = _mm256_broadcast_sd(A + kk + 5*lda);
    register __m256d a6x = _mm256_broadcast_sd(A + kk + 6*lda);
    register __m256d a7x = _mm256_broadcast_sd(A + kk + 7*lda);
    register __m256d b = My_mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
    c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
  }
  My_mm256_storeu_pd(C + lda*0, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + lda*1, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda*2, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + lda*3, c30_c31_c32_c33);
  My_mm256_storeu_pd(C + lda*4, c40_c41_c42_c43);
  My_mm256_storeu_pd(C + lda*5, c50_c51_c52_c53);
  My_mm256_storeu_pd(C + lda*6, c60_c61_c62_c63);
  My_mm256_storeu_pd(C + lda*7, c70_c71_c72_c73);
}

static void do_block_7x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C + lda*0);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda*1);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + lda*2);
  register __m256d c30_c31_c32_c33 = My_mm256_loadu_pd(C + lda*3);
  register __m256d c40_c41_c42_c43 = My_mm256_loadu_pd(C + lda*4);
  register __m256d c50_c51_c52_c53 = My_mm256_loadu_pd(C + lda*5);
  register __m256d c60_c61_c62_c63 = My_mm256_loadu_pd(C + lda*6);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A + kk + 4*lda);
    register __m256d a5x = _mm256_broadcast_sd(A + kk + 5*lda);
    register __m256d a6x = _mm256_broadcast_sd(A + kk + 6*lda);
    register __m256d b = My_mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  }
  My_mm256_storeu_pd(C + lda*0, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + lda*1, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda*2, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + lda*3, c30_c31_c32_c33);
  My_mm256_storeu_pd(C + lda*4, c40_c41_c42_c43);
  My_mm256_storeu_pd(C + lda*5, c50_c51_c52_c53);
  My_mm256_storeu_pd(C + lda*6, c60_c61_c62_c63);
}

static void do_block_6x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + lda*2);
  register __m256d c30_c31_c32_c33 = My_mm256_loadu_pd(C + lda*3);
  register __m256d c40_c41_c42_c43 = My_mm256_loadu_pd(C + lda*4);
  register __m256d c50_c51_c52_c53 = My_mm256_loadu_pd(C + lda*5);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A + kk + 4*lda);
    register __m256d a5x = _mm256_broadcast_sd(A + kk + 5*lda);
    register __m256d b = My_mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda*2, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + lda*3, c30_c31_c32_c33);
  My_mm256_storeu_pd(C + lda*4, c40_c41_c42_c43);
  My_mm256_storeu_pd(C + lda*5, c50_c51_c52_c53);
}

static void do_block_5x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + lda*2);
  register __m256d c30_c31_c32_c33 = My_mm256_loadu_pd(C + lda*3);
  register __m256d c40_c41_c42_c43 = My_mm256_loadu_pd(C + lda*4);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A + kk + 4*lda);
    register __m256d b = My_mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda*2, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + lda*3, c30_c31_c32_c33);
  My_mm256_storeu_pd(C + lda*4, c40_c41_c42_c43);
}

static void do_block_4x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + lda*2);
  register __m256d c30_c31_c32_c33 = My_mm256_loadu_pd(C + lda*3);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3*lda);
    register __m256d b = My_mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda*2, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + lda*3, c30_c31_c32_c33);
}

static void do_block_3x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + lda*2);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d b = My_mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda*2, c20_c21_c22_c23);
}

static void do_block_2x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d b = My_mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
}

static void do_block_1x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d b = My_mm256_loadu_pd(B + kk*lda);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
}

static void do_block_5x8(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c14_c15_c16_c17 = My_mm256_loadu_pd(C + lda + VECTOR_SIZE);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + 2*lda);
  register __m256d c24_c25_c26_c27 = My_mm256_loadu_pd(C + 2*lda + VECTOR_SIZE);
  register __m256d c30_c31_c32_c33 = My_mm256_loadu_pd(C + 3*lda);
  register __m256d c34_c35_c36_c37 = My_mm256_loadu_pd(C + 3*lda + VECTOR_SIZE);
  register __m256d c40_c41_c42_c43 = My_mm256_loadu_pd(C + 4*lda);
  register __m256d c44_c45_c46_c47 = My_mm256_loadu_pd(C + 4*lda + VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A + kk + 4*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b0, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b0, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b0, c40_c41_c42_c43);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
    c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b1, c14_c15_c16_c17);
    c24_c25_c26_c27 = _mm256_fmadd_pd(a2x, b1, c24_c25_c26_c27);
    c34_c35_c36_c37 = _mm256_fmadd_pd(a3x, b1, c34_c35_c36_c37);
    c44_c45_c46_c47 = _mm256_fmadd_pd(a4x, b1, c44_c45_c46_c47);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda + VECTOR_SIZE, c14_c15_c16_c17);
  My_mm256_storeu_pd(C + 2*lda, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + 2*lda + VECTOR_SIZE, c24_c25_c26_c27);
  My_mm256_storeu_pd(C + 3*lda, c30_c31_c32_c33);
  My_mm256_storeu_pd(C + 3*lda + VECTOR_SIZE, c34_c35_c36_c37);
  My_mm256_storeu_pd(C + 4*lda, c40_c41_c42_c43);
  My_mm256_storeu_pd(C + 4*lda + VECTOR_SIZE, c44_c45_c46_c47);
}

static void do_block_4x8(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c14_c15_c16_c17 = My_mm256_loadu_pd(C + lda + VECTOR_SIZE);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + 2*lda);
  register __m256d c24_c25_c26_c27 = My_mm256_loadu_pd(C + 2*lda + VECTOR_SIZE);
  register __m256d c30_c31_c32_c33 = My_mm256_loadu_pd(C + 3*lda);
  register __m256d c34_c35_c36_c37 = My_mm256_loadu_pd(C + 3*lda + VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b0, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b0, c30_c31_c32_c33);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
    c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b1, c14_c15_c16_c17);
    c24_c25_c26_c27 = _mm256_fmadd_pd(a2x, b1, c24_c25_c26_c27);
    c34_c35_c36_c37 = _mm256_fmadd_pd(a3x, b1, c34_c35_c36_c37);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda + VECTOR_SIZE, c14_c15_c16_c17);
  My_mm256_storeu_pd(C + 2*lda, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + 2*lda + VECTOR_SIZE, c24_c25_c26_c27);
  My_mm256_storeu_pd(C + 3*lda, c30_c31_c32_c33);
  My_mm256_storeu_pd(C + 3*lda + VECTOR_SIZE, c34_c35_c36_c37);
}

static void do_block_3x8(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c14_c15_c16_c17 = My_mm256_loadu_pd(C + lda + VECTOR_SIZE);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + 2*lda);
  register __m256d c24_c25_c26_c27 = My_mm256_loadu_pd(C + 2*lda + VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b0, c20_c21_c22_c23);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
    c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b1, c14_c15_c16_c17);
    c24_c25_c26_c27 = _mm256_fmadd_pd(a2x, b1, c24_c25_c26_c27);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda + VECTOR_SIZE, c14_c15_c16_c17);
  My_mm256_storeu_pd(C + 2*lda, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + 2*lda + VECTOR_SIZE, c24_c25_c26_c27);
}

static void do_block_2x8(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c14_c15_c16_c17 = My_mm256_loadu_pd(C + lda + VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0, c10_c11_c12_c13);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
    c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b1, c14_c15_c16_c17);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda + VECTOR_SIZE, c14_c15_c16_c17);
}

static void do_block_1x8(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
}

static void do_block_4x12(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  register __m256d c08_c09_c10_c11 = My_mm256_loadu_pd(C + 2*VECTOR_SIZE);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c14_c15_c16_c17 = My_mm256_loadu_pd(C + lda + VECTOR_SIZE);
  register __m256d c18_c19_c110_c111 = My_mm256_loadu_pd(C + lda + 2*VECTOR_SIZE);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + 2*lda);
  register __m256d c24_c25_c26_c27 = My_mm256_loadu_pd(C + 2*lda + VECTOR_SIZE);
  register __m256d c28_c29_c210_c211 = My_mm256_loadu_pd(C + 2*lda + 2*VECTOR_SIZE);
  register __m256d c30_c31_c32_c33 = My_mm256_loadu_pd(C + 3*lda);
  register __m256d c34_c35_c36_c37 = My_mm256_loadu_pd(C + 3*lda + VECTOR_SIZE);
  register __m256d c38_c39_c310_c311 = My_mm256_loadu_pd(C + 3*lda + 2*VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    register __m256d b2 = My_mm256_loadu_pd(B + kk*lda + 2*VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
    c08_c09_c10_c11 = _mm256_fmadd_pd(a0x, b2, c08_c09_c10_c11);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0, c10_c11_c12_c13);
    c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b1, c14_c15_c16_c17);
    c18_c19_c110_c111 = _mm256_fmadd_pd(a1x, b2, c18_c19_c110_c111);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b0, c20_c21_c22_c23);
    c24_c25_c26_c27 = _mm256_fmadd_pd(a2x, b1, c24_c25_c26_c27);
    c28_c29_c210_c211 = _mm256_fmadd_pd(a2x, b2, c28_c29_c210_c211);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b0, c30_c31_c32_c33);
    c34_c35_c36_c37 = _mm256_fmadd_pd(a3x, b1, c34_c35_c36_c37);
    c38_c39_c310_c311 = _mm256_fmadd_pd(a3x, b2, c38_c39_c310_c311);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
  My_mm256_storeu_pd(C + 2*VECTOR_SIZE, c08_c09_c10_c11);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda + VECTOR_SIZE, c14_c15_c16_c17);
  My_mm256_storeu_pd(C + lda + 2*VECTOR_SIZE, c18_c19_c110_c111);
  My_mm256_storeu_pd(C + 2*lda, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + 2*lda + VECTOR_SIZE, c24_c25_c26_c27);
  My_mm256_storeu_pd(C + 2*lda + 2*VECTOR_SIZE, c28_c29_c210_c211);
  My_mm256_storeu_pd(C + 3*lda, c30_c31_c32_c33);
  My_mm256_storeu_pd(C + 3*lda + VECTOR_SIZE, c34_c35_c36_c37);
  My_mm256_storeu_pd(C + 3*lda + 2*VECTOR_SIZE, c38_c39_c310_c311);
}


static void do_block_3x12(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  register __m256d c08_c09_c10_c11 = My_mm256_loadu_pd(C + 2*VECTOR_SIZE);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c14_c15_c16_c17 = My_mm256_loadu_pd(C + lda + VECTOR_SIZE);
  register __m256d c18_c19_c110_c111 = My_mm256_loadu_pd(C + lda + 2*VECTOR_SIZE);
  register __m256d c20_c21_c22_c23 = My_mm256_loadu_pd(C + 2*lda);
  register __m256d c24_c25_c26_c27 = My_mm256_loadu_pd(C + 2*lda + VECTOR_SIZE);
  register __m256d c28_c29_c210_c211 = My_mm256_loadu_pd(C + 2*lda + 2*VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    register __m256d b2 = My_mm256_loadu_pd(B + kk*lda + 2*VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
    c08_c09_c10_c11 = _mm256_fmadd_pd(a0x, b2, c08_c09_c10_c11);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0, c10_c11_c12_c13);
    c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b1, c14_c15_c16_c17);
    c18_c19_c110_c111 = _mm256_fmadd_pd(a1x, b2, c18_c19_c110_c111);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b0, c20_c21_c22_c23);
    c24_c25_c26_c27 = _mm256_fmadd_pd(a2x, b1, c24_c25_c26_c27);
    c28_c29_c210_c211 = _mm256_fmadd_pd(a2x, b2, c28_c29_c210_c211);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
  My_mm256_storeu_pd(C + 2*VECTOR_SIZE, c08_c09_c10_c11);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda + VECTOR_SIZE, c14_c15_c16_c17);
  My_mm256_storeu_pd(C + lda + 2*VECTOR_SIZE, c18_c19_c110_c111);
  My_mm256_storeu_pd(C + 2*lda, c20_c21_c22_c23);
  My_mm256_storeu_pd(C + 2*lda + VECTOR_SIZE, c24_c25_c26_c27);
  My_mm256_storeu_pd(C + 2*lda + 2*VECTOR_SIZE, c28_c29_c210_c211);
}

static void do_block_2x12(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  register __m256d c08_c09_c10_c11 = My_mm256_loadu_pd(C + 2*VECTOR_SIZE);
  register __m256d c10_c11_c12_c13 = My_mm256_loadu_pd(C + lda);
  register __m256d c14_c15_c16_c17 = My_mm256_loadu_pd(C + lda + VECTOR_SIZE);
  register __m256d c18_c19_c110_c111 = My_mm256_loadu_pd(C + lda + 2*VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    register __m256d b2 = My_mm256_loadu_pd(B + kk*lda + 2*VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
    c08_c09_c10_c11 = _mm256_fmadd_pd(a0x, b2, c08_c09_c10_c11);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0, c10_c11_c12_c13);
    c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b1, c14_c15_c16_c17);
    c18_c19_c110_c111 = _mm256_fmadd_pd(a1x, b2, c18_c19_c110_c111);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
  My_mm256_storeu_pd(C + 2*VECTOR_SIZE, c08_c09_c10_c11);
  My_mm256_storeu_pd(C + lda, c10_c11_c12_c13);
  My_mm256_storeu_pd(C + lda + VECTOR_SIZE, c14_c15_c16_c17);
  My_mm256_storeu_pd(C + lda + 2*VECTOR_SIZE, c18_c19_c110_c111);
}

static void do_block_1x12(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = My_mm256_loadu_pd(C);
  register __m256d c04_c05_c06_c07 = My_mm256_loadu_pd(C + VECTOR_SIZE);
  register __m256d c08_c09_c010_c011 = My_mm256_loadu_pd(C + 2*VECTOR_SIZE);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d b0 = My_mm256_loadu_pd(B + kk*lda);
    register __m256d b1 = My_mm256_loadu_pd(B + kk*lda + VECTOR_SIZE);
    register __m256d b2 = My_mm256_loadu_pd(B + kk*lda + 2*VECTOR_SIZE);
    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0, c00_c01_c02_c03);
    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b1, c04_c05_c06_c07);
    c08_c09_c010_c011 = _mm256_fmadd_pd(a0x, b2, c08_c09_c010_c011);
  }
  My_mm256_storeu_pd(C, c00_c01_c02_c03);
  My_mm256_storeu_pd(C + VECTOR_SIZE, c04_c05_c06_c07);
  My_mm256_storeu_pd(C + 2*VECTOR_SIZE, c08_c09_c010_c011);
}

static void do_block_simd_remainder(int lda, int K, int N_remain, double* A, double* B, double* C) {
  // N_remain = number of remaining columns that is less than VLEN
  for (int n = 0; n < N_remain; n++) {
    double c = C[n];
    for (int i = 0; i < K; i++) {
      c += A[i]*B[i*lda + n];
    }
    C[n] = c;
  }
}

#ifdef SIMD_8x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 8) {
    int M2 = min(8, M - i);
    switch (M2) {
      case 8:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_8x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+5)*lda, B + j, C + (i+5)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+6)*lda, B + j, C + (i+6)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+7)*lda, B + j, C + (i+7)*lda + j);
          }
        }
        break;
      case 7:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_7x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+5)*lda, B + j, C + (i+5)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+6)*lda, B + j, C + (i+6)*lda + j);
          }
        }
        break;
      case 6:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_6x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+5)*lda, B + j, C + (i+5)*lda + j);
          }
        }
        break;
      case 5:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_5x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
          }
        }
        break;
      case 4:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
          }
        }
        break;
      case 3:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          }
        }
        break;
      case 2:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          }
        }
        break;
      case 1:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          }
        }
        break;
    }
  }
}
#endif

#ifdef SIMD_7x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 7) {
    int M2 = min(7, M - i);
    switch (M2) {
      case 7:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_7x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+5)*lda, B + j, C + (i+5)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+6)*lda, B + j, C + (i+6)*lda + j);
          }
        }
        break;
      case 6:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_6x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+5)*lda, B + j, C + (i+5)*lda + j);
          }
        }
        break;
      case 5:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_5x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
          }
        }
        break;
      case 4:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
          }
        }
        break;
      case 3:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          }
        }
        break;
      case 2:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          }
        }
        break;
      case 1:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          }
        }
        break;
    }
  }
}
#endif


#ifdef SIMD_6x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 6) {
    int M2 = min(6, M - i);
    switch (M2) {
      case 6:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_6x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+5)*lda, B + j, C + (i+5)*lda + j);
          }
        }
        break;
      case 5:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_5x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
          }
        }
        break;
      case 4:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
          }
        }
        break;
      case 3:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          }
        }
        break;
      case 2:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          }
        }
        break;
      case 1:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          }
        }
        break;
    }
  }
}
#endif

#ifdef SIMD_5x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 5) {
    int M2 = min(5, M - i);
    switch (M2) {
      case 5:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_5x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
          }
        }
        break;
      case 4:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
          }
        }
        break;
      case 3:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          }
        }
        break;
      case 2:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          }
        }
        break;
      case 1:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          }
        }
        break;
    }
  }
}
#endif

#ifdef SIMD_4x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 4) {
    int M2 = min(4, M - i);
    switch (M2) {
      case 4:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
          }
        }
        break;
      case 3:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          }
        }
        break;
      case 2:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          }
        }
        break;
      case 1:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          }
        }
        break;
    }
  }
}
#endif

#ifdef SIMD_3x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 3) {
    int M2 = min(3, M - i);
    switch (M2) {
      case 3:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          }
        }
        break;
      case 2:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
            do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          }
        }
        break;
      case 1:
        for (int j = 0; j < N; j += VECTOR_SIZE) {
          int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
          if (N2 == VECTOR_SIZE) {
            // Multiples of VECTOR_SIZE
            do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          } else {
            // Less than VECTOR_SIZE
            do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          }
        }
        break;
    }
  }
}
#endif

#ifdef SIMD_2x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 2) {
    int M2 = min(2, M - i);
    switch (M2) {
    case 2:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += VECTOR_SIZE) {
        int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
        }
      }
      break;
    case 1:
      for (int j = 0; j < N; j += VECTOR_SIZE) {
        int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
        }
      }
      break;
    }
  }
}
#endif

#ifdef SIMD_1x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i++) {
    int M2 = min(2, M - i);
    for (int j = 0; j < N; j += VECTOR_SIZE) {
      int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
      if (N2 == VECTOR_SIZE) {
        // Multiples of VECTOR_SIZE
        do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
      } else {
        // Less than VECTOR_SIZE
        do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
      }
    }
  }
}
#endif

#ifdef SIMD_1x4
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each row i of A */
  for (int i = 0; i < M; i++) {
    /* For each VLEN columns of B */
    for (int j = 0; j < N; j += VECTOR_SIZE) {
      int N2 = min(VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
      if (N2 == VECTOR_SIZE) {
        // Multiples of VECTOR_SIZE
        do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
      } else {
        // Less than VECTOR_SIZE
        do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
      }
    }
  }
}
#endif

#ifdef SIMD_5x8
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 5) {
    int M2 = min(5, M - i);
    switch (M2) {
    case 5:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_5x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_5x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+3)*lda, B + j + VECTOR_SIZE, C + (i+3)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+4)*lda, B + j + VECTOR_SIZE, C + (i+4)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_5x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+4)*lda, B + j, C + (i+4)*lda + j);
        }
      }
      break;
    case 4:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_4x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+3)*lda, B + j + VECTOR_SIZE, C + (i+3)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
        }
      }
      break;
    case 3:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_3x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
        }
      }
      break;
    case 2:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
        }
      }
      break;
    case 1:
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
        }
      }
      break;
    }
  }
}
#endif

#ifdef SIMD_4x8
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 4) {
    int M2 = min(4, M - i);
    switch (M2) {
    case 4:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_4x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
            do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+3)*lda, B + j + VECTOR_SIZE, C + (i+3)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
        }
      }
      break;
    case 3:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_3x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
        }
      }
      break;
    case 2:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
        }
      }
      break;
    case 1:
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
        }
      }
      break;
    }
  }
}
#endif

#ifdef SIMD_3x8
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 3) {
    int M2 = min(3, M - i);
    switch (M2) {
    case 3:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_3x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
        }
      }
      break;
    case 2:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
        }
      }
      break;
    case 1:
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
        }
      }
      break;
    }
  }
}
#endif

#ifdef SIMD_2x8
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i += 2) {
    int M2 = min(2, M - i);
    switch (M2) {
    case 2:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
        }
      }
      break;
    case 1:
      for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
        int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 2 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
        }
      }
      break;
    }
  }
}
#endif

#ifdef SIMD_1x8
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i++) {
    int M2 = min(2, M - i);

    for (int j = 0; j < N; j += 2 * VECTOR_SIZE) {
      int N2 = min(2 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
      if (N2 == 2 * VECTOR_SIZE) {
        do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
      } else if (N2 > VECTOR_SIZE) {
        do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
      } else if (N2 == VECTOR_SIZE) {
        // Multiples of VECTOR_SIZE
        do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
      } else {
        // Less than VECTOR_SIZE
        do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
      }
    }
  }
}
#endif

#ifdef SIMD_4x12
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i+=4) {
    int M2 = min(4, M - i);
    switch (M2) {
    case 4:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_4x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_4x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+1)*lda, B + j + 2*VECTOR_SIZE, C + (i+1)*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+2)*lda, B + j + 2*VECTOR_SIZE, C + (i+2)*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+3)*lda, B + j + 2*VECTOR_SIZE, C + (i+3)*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_4x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+3)*lda, B + j + VECTOR_SIZE, C + (i+3)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_4x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+3)*lda, B + j, C + (i+3)*lda + j);
        }
      }
      break;
    case 3:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_3x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_3x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+1)*lda, B + j + 2*VECTOR_SIZE, C + (i+1)*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+2)*lda, B + j + 2*VECTOR_SIZE, C + (i+2)*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_3x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
        }
      }
      break;
    case 2:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_2x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+1)*lda, B + j + 2*VECTOR_SIZE, C + (i+1)*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
        }
      }
      break;
    case 1:
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_1x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
        }
      }
      break;
    }
  }
}
#endif

#ifdef SIMD_3x12
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i+=3) {
    int M2 = min(3, M - i);
    switch (M2) {
    case 3:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_3x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_3x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+1)*lda, B + j + 2*VECTOR_SIZE, C + (i+1)*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+2)*lda, B + j + 2*VECTOR_SIZE, C + (i+2)*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_3x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+2)*lda, B + j + VECTOR_SIZE, C + (i+2)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_3x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+2)*lda, B + j, C + (i+2)*lda + j);
        }
      }
      break;
    case 2:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_2x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+1)*lda, B + j + 2*VECTOR_SIZE, C + (i+1)*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
        }
      }
      break;
    case 1:
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_1x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
        }
      }
      break;
    }
  }
}
#endif

#ifdef SIMD_2x12
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i+=2) {
    int M2 = min(2, M - i);
    switch (M2) {
    case 2:
      /* For each VLEN columns of B */
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_2x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + (i+1)*lda, B + j + 2*VECTOR_SIZE, C + (i+1)*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_2x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + (i+1)*lda, B + j + VECTOR_SIZE, C + (i+1)*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_2x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2, A + (i+1)*lda, B + j, C + (i+1)*lda + j);
        }
      }
      break;
    case 1:
      for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
        int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
        if (N2 == 3 * VECTOR_SIZE) {
          do_block_1x12(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
        } else if (N2 == 2 * VECTOR_SIZE) {
          do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else if (N2 > VECTOR_SIZE) {
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
          do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
        } else if (N2 == VECTOR_SIZE) {
          // Multiples of VECTOR_SIZE
          do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        } else {
          // Less than VECTOR_SIZE
          do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
        }
      }
      break;
    }
  }
}
#endif

#ifdef SIMD_1x12
static void do_block_simd(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each two rows i of A */
  for (int i = 0; i < M; i++) {
    // int M2 = min(2, M - i);

    for (int j = 0; j < N; j += 3 * VECTOR_SIZE) {
      int N2 = min(3 * VECTOR_SIZE, N-j); /* Correct block dimensions if block "goes off edge of" the matrix */
      if (N2 == 3 * VECTOR_SIZE) {
        do_block_1x12(lda, K, A + i*lda, B + j, C + i*lda + j);
      } else if (N2 > 2 * VECTOR_SIZE && N2 < 3 * VECTOR_SIZE) {
        do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
        do_block_simd_remainder(lda, K, N2 - 2*VECTOR_SIZE, A + i*lda, B + j + 2*VECTOR_SIZE, C + i*lda + j + 2*VECTOR_SIZE);
      } else if (N2 == 2 * VECTOR_SIZE) {
        do_block_1x8(lda, K, A + i*lda, B + j, C + i*lda + j);
      } else if (N2 > VECTOR_SIZE) {
        do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
        do_block_simd_remainder(lda, K, N2 - VECTOR_SIZE, A + i*lda, B + j + VECTOR_SIZE, C + i*lda + j + VECTOR_SIZE);
      } else if (N2 == VECTOR_SIZE) {
        // Multiples of VECTOR_SIZE
        do_block_1x4(lda, K, A + i*lda, B + j, C + i*lda + j);
      } else {
        // Less than VECTOR_SIZE
        do_block_simd_remainder(lda, K, N2, A + i*lda, B + j, C + i*lda + j);
      }
    }
  }
}
#endif

#ifdef LAYOUT
void copy_layout(int lda, int M, int N, double *src, double *dst) {
    int pos = 0;
    for (int i = 0; i < M; i++) {
        memcpy(dst + pos, src + i * lda, N * sizeof(double));
        pos += N;
    }
}

static void copy_back_layout(int lda, int M, int N, double *src, double *dst) {
    int pos = 0;
    for (int i = 0; i < M; i++) {
        memcpy(dst + i * lda, src + pos, N * sizeof(double));
        pos += N;
    }
}

static void do_block_layout(int lda, int M, int N, int K, double *A, double *B, double *C) {
    assert(M <= L1_BLOCK_SIZE_M);
    assert(N <= L1_BLOCK_SIZE_N);
    assert(K <= L1_BLOCK_SIZE_K);
    double As[L1_BLOCK_SIZE_M * L1_BLOCK_SIZE_K];
    double Bs[L1_BLOCK_SIZE_K * L1_BLOCK_SIZE_N];
    double Cs[L1_BLOCK_SIZE_M * L1_BLOCK_SIZE_N];

    copy_layout(lda, M, K, A, As);

    #ifdef TRANSPOSE
    copy_layout(lda, N, K, B, Bs);
    #else
    copy_layout(lda, K, N, B, Bs);
    #endif
    copy_layout(lda, M, N, C, Cs);
    for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            /* Compute C(i,j) */
            double cij = Cs[i*N+j];
            for (int k = 0; k < K; ++k)
            #ifdef TRANSPOSE
                cij += As[i*K+k] * Bs[j*K+k];
            #else
                cij += As[i*K+k] * Bs[k*N+j];
            #endif
            Cs[i*N+j] = cij;
        }
    copy_back_layout(lda, M, N, Cs, C);
}
#endif


static void do_block_1(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each row i of A */
  for (int i = 0; i < M; i += L1_BLOCK_SIZE_M) {
    /* For each block-column of B */
    for (int j = 0; j < N; j += L1_BLOCK_SIZE_N) {
      /* Accumulate block dgemms into block of C */

      for (int k = 0; k < K; k += L1_BLOCK_SIZE_K) {
      	/* Correct block dimensions if block "goes off edge of" the matrix */
      	int M2 = min (L1_BLOCK_SIZE_M, M-i);
      	int N2 = min (L1_BLOCK_SIZE_N, N-j);
      	int K2 = min (L1_BLOCK_SIZE_K, K-k);

        #ifdef LAYOUT
        #ifdef TRANSPOSE
          do_block_layout(lda, M2, N2, K2, A + i*lda + k, B + j*lda + k, C + i*lda + j);
        #else
          do_block_layout(lda, M2, N2, K2, A + i*lda + k, B + k*lda + j, C + i*lda + j);
        #endif
        #endif

        /* Perform individual block dgemm */
        #ifdef TRANSPOSE
        	do_block(lda, M2, N2, K2, A + i*lda + k, B + j*lda + k, C + i*lda + j);
        #else
          #if defined(SIMD_1x4) || defined(SIMD_2x4) || defined(SIMD_3x4) \
          || defined(SIMD_4x4) || defined(SIMD_5x4) || defined(SIMD_6x4) \
          || defined(SIMD_7x4) || defined(SIMD_8x4) \
          || defined(SIMD_1x8) || defined(SIMD_2x8) || defined(SIMD_3x8) \
          || defined(SIMD_4x8) || defined(SIMD_5x8) \
          || defined(SIMD_1x12) || defined(SIMD_2x12) || defined(SIMD_3x12) \
          || defined(SIMD_4x12)
            // No transpose if use SIMD as we are using register tiling to access memory in row major order
            do_block_simd(lda, M2, N2, K2, A + i*lda + k, B + k*lda + j, C + i*lda + j);
          #else
            do_block(lda, M2, N2, K2, A + i*lda + k, B + k*lda + j, C + i*lda + j);
          #endif
        #endif
      }
    }
  }
}

static void do_block_2(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each row i of A */
  for (int i = 0; i < M; i += L2_BLOCK_SIZE_M) {
    /* For each block-column of B */
    for (int j = 0; j < N; j += L2_BLOCK_SIZE_N) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < K; k += L2_BLOCK_SIZE_K) {
      	/* Correct block dimensions if block "goes off edge of" the matrix */
      	int M2 = min (L2_BLOCK_SIZE_M, M-i);
      	int N2 = min (L2_BLOCK_SIZE_N, N-j);
      	int K2 = min (L2_BLOCK_SIZE_K, K-k);

        /* Perform individual block dgemm */
        #ifdef TRANSPOSE
        	do_block_1(lda, M2, N2, K2, A + i*lda + k, B + j*lda + k, C + i*lda + j);
        #else
        	do_block_1(lda, M2, N2, K2, A + i*lda + k, B + k*lda + j, C + i*lda + j);
        #endif
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {

  #ifdef CACHELAYOUT
  double *a, *b, *c;
  int ldn = dealias(lda);
  if (ldn != lda) {
    copy_layout(lda, A, B, C, ldn, &a, &b, &c);
    for (int i = 0; i < lda; i += L3_BLOCK_SIZE_M) {
      for (int j = 0; j < lda; j += L3_BLOCK_SIZE_N) {
        for (int k = 0; k < lda; k += L3_BLOCK_SIZE_K) {
          int M = min (L3_BLOCK_SIZE_M, lda-i);
          int N = min (L3_BLOCK_SIZE_N, lda-j);
          int K = min (L3_BLOCK_SIZE_K, lda-k);
          do_block_2(ldn, M, N, K, a + i*ldn + k, b + k*ldn + j, c + i*ldn + j);
        }
      }
    }
    copy_back_layout(ldn, &a, &b, &c, lda, C);
    return;
  }
  #endif

  #ifdef TRANSPOSE
    for (int i = 0; i < lda; ++i)
      for (int j = i+1; j < lda; ++j) {
          double t = B[i*lda+j];
          B[i*lda+j] = B[j*lda+i];
          B[j*lda+i] = t;
      }
  #endif
  /* For each block-row of A */
  for (int i = 0; i < lda; i += L3_BLOCK_SIZE_M) {
    /* For each block-column of B */
    for (int j = 0; j < lda; j += L3_BLOCK_SIZE_N) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += L3_BLOCK_SIZE_K) {
      	/* Correct block dimensions if block "goes off edge of" the matrix */
      	int M = min (L3_BLOCK_SIZE_M, lda-i);
      	int N = min (L3_BLOCK_SIZE_N, lda-j);
      	int K = min (L3_BLOCK_SIZE_K, lda-k);

        /* Perform individual block dgemm */
        #ifdef TRANSPOSE
        	do_block_2(lda, M, N, K, A + i*lda + k, B + j*lda + k, C + i*lda + j);
        #else
        	do_block_2(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
        #endif
      }
    }
  }
  #if TRANSPOSE
    for (int i = 0; i < lda; ++i) {
      for (int j = i+1; j < lda; ++j) {
          double t = B[i*lda+j];
          B[i*lda+j] = B[j*lda+i];
          B[j*lda+i] = t;
      }
    }
  #endif
}



// /* Single level blocked */
// /* This auxiliary subroutine performs a smaller dgemm operation
//  *  C := C + A * B
//  * where C is M-by-N, A is M-by-K, and B is K-by-N. */
// static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
//   /* For each row i of A */
//   for (int i = 0; i < M; ++i)
//     /* For each column j of B */
//     for (int j = 0; j < N; ++j)
//     {
//       /* Compute C(i,j) */
//       double cij = C[i*lda+j];
//       for (int k = 0; k < K; ++k)
// #ifdef TRANSPOSE
// 	cij += A[i*lda+k] * B[j*lda+k];
// #else
// 	cij += A[i*lda+k] * B[k*lda+j];
// #endif
//       C[i*lda+j] = cij;
//     }
// }
//
// /* This routine performs a dgemm operation
//  *  C := C + A * B
//  * where A, B, and C are lda-by-lda matrices stored in row-major order
//  * On exit, A and B maintain their input values. */
// void square_dgemm(int lda, double* A, double* B, double* C) {
// #ifdef TRANSPOSE
//   for (int i = 0; i < lda; ++i)
//     for (int j = i+1; j < lda; ++j) {
//         double t = B[i*lda+j];
//         B[i*lda+j] = B[j*lda+i];
//         B[j*lda+i] = t;
//     }
// #endif
//   /* For each block-row of A */
//   for (int i = 0; i < lda; i += BLOCK_SIZE)
//     /* For each block-column of B */
//     for (int j = 0; j < lda; j += BLOCK_SIZE)
//       /* Accumulate block dgemms into block of C */
//       for (int k = 0; k < lda; k += BLOCK_SIZE)
//       {
// 	/* Correct block dimensions if block "goes off edge of" the matrix */
// 	int M = min (BLOCK_SIZE, lda-i);
// 	int N = min (BLOCK_SIZE, lda-j);
// 	int K = min (BLOCK_SIZE, lda-k);
//
// 	/* Perform individual block dgemm */
// #ifdef TRANSPOSE
// 	do_block(lda, M, N, K, A + i*lda + k, B + j*lda + k, C + i*lda + j);
// #else
// 	do_block(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
// #endif
//       }
// #if TRANSPOSE
//   for (int i = 0; i < lda; ++i)
//     for (int j = i+1; j < lda; ++j) {
//         double t = B[i*lda+j];
//         B[i*lda+j] = B[j*lda+i];
//         B[j*lda+i] = t;
//     }
// #endif
// }
