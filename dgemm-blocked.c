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

const char* dgemm_desc = "Multilevel blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 37
#define L1_BLOCK_SIZE 34
#define L2_BLOCK_SIZE 102
#define L3_BLOCK_SIZE 1122
#define VECTOR_SIZE 4
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

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


static void do_block_1x4(int lda, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  // register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C + lda);
  for (int kk = 0; kk < K; kk++) {
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    // register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d b = _mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    // c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  }
  _mm256_storeu_pd(C, c00_c01_c02_c03);
  // _mm256_store_pd(C + lda, c10_c11_c12_c13);
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



static void do_block_1(int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each row i of A */
  for (int i = 0; i < M; i += L1_BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < N; j += L1_BLOCK_SIZE) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < K; k += L1_BLOCK_SIZE) {
      	/* Correct block dimensions if block "goes off edge of" the matrix */
      	int M2 = min (L1_BLOCK_SIZE, M-i);
      	int N2 = min (L1_BLOCK_SIZE, N-j);
      	int K2 = min (L1_BLOCK_SIZE, K-k);

        /* Perform individual block dgemm */
        #ifdef TRANSPOSE
        	do_block(lda, M2, N2, K2, A + i*lda + k, B + j*lda + k, C + i*lda + j);
        #else
          #ifdef SIMD
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
  for (int i = 0; i < M; i += L2_BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < N; j += L2_BLOCK_SIZE) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < K; k += L2_BLOCK_SIZE) {
      	/* Correct block dimensions if block "goes off edge of" the matrix */
      	int M2 = min (L2_BLOCK_SIZE, M-i);
      	int N2 = min (L2_BLOCK_SIZE, N-j);
      	int K2 = min (L2_BLOCK_SIZE, K-k);

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
  static int print_guard = 1;
  if (print_guard) {
      print_guard = 0;
      printf("L1 Block Size: %d\n", L1_BLOCK_SIZE);
      printf("L2 Block Size: %d\n", L2_BLOCK_SIZE);
      printf("L3 Block Size: %d\n", L3_BLOCK_SIZE);
  }

  #ifdef TRANSPOSE
    for (int i = 0; i < lda; ++i)
      for (int j = i+1; j < lda; ++j) {
          double t = B[i*lda+j];
          B[i*lda+j] = B[j*lda+i];
          B[j*lda+i] = t;
      }
  #endif
  /* For each block-row of A */
  for (int i = 0; i < lda; i += L3_BLOCK_SIZE) {
    /* For each block-column of B */
    for (int j = 0; j < lda; j += L3_BLOCK_SIZE) {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += L3_BLOCK_SIZE) {
      	/* Correct block dimensions if block "goes off edge of" the matrix */
      	int M = min (L3_BLOCK_SIZE, lda-i);
      	int N = min (L3_BLOCK_SIZE, lda-j);
      	int K = min (L3_BLOCK_SIZE, lda-k);

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
