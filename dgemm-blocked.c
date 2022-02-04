#include <immintrin.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 152
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define SIMDBLOCK 8
#define A(i,j, size) (A[i * size + j]) 

double* col_to_row_major(double* M, int lda, int size);
double* cache_align_col(double* B, int size);
void copy_mat(double* C, double* M, int lda, int size);

void simd_8x8_kernel(double* A, double* B, double*  C, int size) {
    __m512d b0, b1, b2, b3, b4, b5, b6, b7;
    __m512d c0, c1, c2, c3, c4, c5, c6, c7;

    b0 = _mm512_load_pd(B);
    b1 = _mm512_load_pd(B + size);
    b2 = _mm512_load_pd(B + 2 * size);
    b3 = _mm512_load_pd(B + 3 * size);
    b4 = _mm512_load_pd(B + 4 * size);
    b5 = _mm512_load_pd(B + 5 * size);
    b6 = _mm512_load_pd(B + 6 * size);
    b7 = _mm512_load_pd(B + 7 * size);

    c0 = _mm512_load_pd(C);
    c1 = _mm512_load_pd(C + size);
    c2 = _mm512_load_pd(C + 2 * size);
    c3 = _mm512_load_pd(C + 3 * size);
    c4 = _mm512_load_pd(C + 4 * size);
    c5 = _mm512_load_pd(C + 5 * size);
    c6 = _mm512_load_pd(C + 6 * size);
    c7 = _mm512_load_pd(C + 7 * size);

    __m512d a_val;
    a_val = _mm512_set1_pd(A(0, 0, size));
    c0 = _mm512_fmadd_pd(a_val, b0, c0);
    a_val = _mm512_set1_pd(A(0, 1, size));
    c0 = _mm512_fmadd_pd(a_val, b1, c0);
    a_val = _mm512_set1_pd(A(0, 2, size));
    c0 = _mm512_fmadd_pd(a_val, b2, c0);
    a_val = _mm512_set1_pd(A(0, 3, size));
    c0 = _mm512_fmadd_pd(a_val, b3, c0);
    a_val = _mm512_set1_pd(A(0, 4, size));
    c0  = _mm512_fmadd_pd(a_val, b4, c0);
    a_val = _mm512_set1_pd(A(0, 5, size));
    c0 = _mm512_fmadd_pd(a_val, b5, c0);
    a_val = _mm512_set1_pd(A(0, 6, size));
    c0 = _mm512_fmadd_pd(a_val, b6, c0);
    a_val = _mm512_set1_pd(A(0, 7, size));
    c0 = _mm512_fmadd_pd(a_val, b7, c0);

    a_val = _mm512_set1_pd(A(1, 0, size));
    c1 = _mm512_fmadd_pd(a_val, b0, c1);
    a_val = _mm512_set1_pd(A(1, 1, size));
    c1 = _mm512_fmadd_pd(a_val, b1, c1);
    a_val = _mm512_set1_pd(A(1, 2, size));
    c1 = _mm512_fmadd_pd(a_val, b2, c1);
    a_val = _mm512_set1_pd(A(1, 3, size));
    c1 = _mm512_fmadd_pd(a_val, b3, c1);
    a_val = _mm512_set1_pd(A(1, 4, size));
    c1  = _mm512_fmadd_pd(a_val, b4, c1);
    a_val = _mm512_set1_pd(A(1, 5, size));
    c1 = _mm512_fmadd_pd(a_val, b5, c1);
    a_val = _mm512_set1_pd(A(1, 6, size));
    c1 = _mm512_fmadd_pd(a_val, b6, c1);
    a_val = _mm512_set1_pd(A(1, 7, size));
    c1 = _mm512_fmadd_pd(a_val, b7, c1);

    a_val = _mm512_set1_pd(A(2, 0, size));
    c2 = _mm512_fmadd_pd(a_val, b0, c2);
    a_val = _mm512_set1_pd(A(2, 1, size));
    c2 = _mm512_fmadd_pd(a_val, b1, c2);
    a_val = _mm512_set1_pd(A(2, 2, size));
    c2 = _mm512_fmadd_pd(a_val, b2, c2);
    a_val = _mm512_set1_pd(A(2, 3, size));
    c2 = _mm512_fmadd_pd(a_val, b3, c2);
    a_val = _mm512_set1_pd(A(2, 4, size));
    c2  = _mm512_fmadd_pd(a_val, b4, c2);
    a_val = _mm512_set1_pd(A(2, 5, size));
    c2 = _mm512_fmadd_pd(a_val, b5, c2);
    a_val = _mm512_set1_pd(A(2, 6, size));
    c2 = _mm512_fmadd_pd(a_val, b6, c2);
    a_val = _mm512_set1_pd(A(2, 7, size));
    c2 = _mm512_fmadd_pd(a_val, b7, c2);

    a_val = _mm512_set1_pd(A(3, 0, size));
    c3 = _mm512_fmadd_pd(a_val, b0, c3);
    a_val = _mm512_set1_pd(A(3, 1, size));
    c3 = _mm512_fmadd_pd(a_val, b1, c3);
    a_val = _mm512_set1_pd(A(3, 2, size));
    c3 = _mm512_fmadd_pd(a_val, b2, c3);
    a_val = _mm512_set1_pd(A(3, 3, size));
    c3 = _mm512_fmadd_pd(a_val, b3, c3);
    a_val = _mm512_set1_pd(A(3, 4, size));
    c3  = _mm512_fmadd_pd(a_val, b4, c3);
    a_val = _mm512_set1_pd(A(3, 5, size));
    c3 = _mm512_fmadd_pd(a_val, b5, c3);
    a_val = _mm512_set1_pd(A(3, 6, size));
    c3 = _mm512_fmadd_pd(a_val, b6, c3);
    a_val = _mm512_set1_pd(A(3, 7, size));
    c3 = _mm512_fmadd_pd(a_val, b7, c3);

    a_val = _mm512_set1_pd(A(4, 0, size));
    c4 = _mm512_fmadd_pd(a_val, b0, c4);
    a_val = _mm512_set1_pd(A(4, 1, size));
    c4 = _mm512_fmadd_pd(a_val, b1, c4);
    a_val = _mm512_set1_pd(A(4, 2, size));
    c4 = _mm512_fmadd_pd(a_val, b2, c4);
    a_val = _mm512_set1_pd(A(4, 3, size));
    c4 = _mm512_fmadd_pd(a_val, b3, c4);
    a_val = _mm512_set1_pd(A(4, 4, size));
    c4  = _mm512_fmadd_pd(a_val, b4, c4);
    a_val = _mm512_set1_pd(A(4, 5, size));
    c4 = _mm512_fmadd_pd(a_val, b5, c4);
    a_val = _mm512_set1_pd(A(4, 6, size));
    c4 = _mm512_fmadd_pd(a_val, b6, c4);
    a_val = _mm512_set1_pd(A(4, 7, size));
    c4 = _mm512_fmadd_pd(a_val, b7, c4);

    a_val = _mm512_set1_pd(A(5, 0, size));
    c5 = _mm512_fmadd_pd(a_val, b0, c5);
    a_val = _mm512_set1_pd(A(5, 1, size));
    c5 = _mm512_fmadd_pd(a_val, b1, c5);
    a_val = _mm512_set1_pd(A(5, 2, size));
    c5 = _mm512_fmadd_pd(a_val, b2, c5);
    a_val = _mm512_set1_pd(A(5, 3, size));
    c5 = _mm512_fmadd_pd(a_val, b3, c5);
    a_val = _mm512_set1_pd(A(5, 4, size));
    c5  = _mm512_fmadd_pd(a_val, b4, c5);
    a_val = _mm512_set1_pd(A(5, 5, size));
    c5 = _mm512_fmadd_pd(a_val, b5, c5);
    a_val = _mm512_set1_pd(A(5, 6, size));
    c5 = _mm512_fmadd_pd(a_val, b6, c5);
    a_val = _mm512_set1_pd(A(5, 7, size));
    c5 = _mm512_fmadd_pd(a_val, b7, c5);

    a_val = _mm512_set1_pd(A(6, 0, size));
    c6 = _mm512_fmadd_pd(a_val, b0, c6);
    a_val = _mm512_set1_pd(A(6, 1, size));
    c6 = _mm512_fmadd_pd(a_val, b1, c6);
    a_val = _mm512_set1_pd(A(6, 2, size));
    c6 = _mm512_fmadd_pd(a_val, b2, c6);
    a_val = _mm512_set1_pd(A(6, 3, size));
    c6 = _mm512_fmadd_pd(a_val, b3, c6);
    a_val = _mm512_set1_pd(A(6, 4, size));
    c6  = _mm512_fmadd_pd(a_val, b4, c6);
    a_val = _mm512_set1_pd(A(6, 5, size));
    c6 = _mm512_fmadd_pd(a_val, b5, c6);
    a_val = _mm512_set1_pd(A(6, 6, size));
    c6 = _mm512_fmadd_pd(a_val, b6, c6);
    a_val = _mm512_set1_pd(A(6, 7, size));
    c6 = _mm512_fmadd_pd(a_val, b7, c6);

    a_val = _mm512_set1_pd(A(7, 0, size));
    c7 = _mm512_fmadd_pd(a_val, b0, c7);
    a_val = _mm512_set1_pd(A(7, 1, size));
    c7 = _mm512_fmadd_pd(a_val, b1, c7);
    a_val = _mm512_set1_pd(A(7, 2, size));
    c7 = _mm512_fmadd_pd(a_val, b2, c7);
    a_val = _mm512_set1_pd(A(7, 3, size));
    c7 = _mm512_fmadd_pd(a_val, b3, c7);
    a_val = _mm512_set1_pd(A(7, 4, size));
    c7  = _mm512_fmadd_pd(a_val, b4, c7);
    a_val = _mm512_set1_pd(A(7, 5, size));
    c7 = _mm512_fmadd_pd(a_val, b5, c7);
    a_val = _mm512_set1_pd(A(7, 6, size));
    c7 = _mm512_fmadd_pd(a_val, b6, c7);
    a_val = _mm512_set1_pd(A(7, 7, size));
    c7 = _mm512_fmadd_pd(a_val, b7, c7);

    _mm512_store_pd(C, c0);
    _mm512_store_pd(C + size, c1);
    _mm512_store_pd(C + 2 * size, c2);
    _mm512_store_pd(C + 3 * size, c3);
    _mm512_store_pd(C + 4 * size, c4);
    _mm512_store_pd(C + 5 * size, c5);
    _mm512_store_pd(C + 6 * size, c6);
    _mm512_store_pd(C + 7 * size, c7);
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {    
    // For each row i of A
    
        for (int i = 0; i < M; i += SIMDBLOCK) {
            for (int j = 0; j < N; j += SIMDBLOCK) {
                for (int k = 0; k < K; k += SIMDBLOCK) {
            
            // For each block-column of B
            
                // Accumulate block dgemms into block of C
                simd_8x8_kernel(A + i * lda + k, B + k * lda + j, C + i * lda + j, lda);
            }
        }
    }
}

void print(int size, double *A) {
    fprintf(stderr, "%d \n", size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(stderr, "%f ", A(i,j, size));
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "*************\n");
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {

    int size;
    if (lda % SIMDBLOCK != 0) {
        size = lda + SIMDBLOCK - (lda % SIMDBLOCK);
    } else {
        size = lda;
    }
    double* copyA = col_to_row_major(A, lda, size);
    double* copyB = col_to_row_major(B, lda, size);
    double* copyC = col_to_row_major(C, lda, size);
    // [1,2,3,4,5,6,7,8,9]


    for (int i = 0; i < size; i += BLOCK_SIZE) {
    
    // For each block-row of A
     
        for (int j = 0; j < size; j += BLOCK_SIZE) {
            for (int k = 0; k < size; k += BLOCK_SIZE) {
           
            // For each block-column of B
            
                    // Perform individual block dgemm
                int M = min(BLOCK_SIZE, size - i);
                int N = min(BLOCK_SIZE, size - j);
                int K = min(BLOCK_SIZE, size - k);
                do_block(size, M, N, K, copyA + i * size + k, copyB + k * size + j, copyC + i * size + j);
            }
        }
    }
    copy_mat(C, copyC, lda, size);
}

double* col_to_row_major(double* M, int lda, int size) {
    //allocate space for copy
    double* copy = (double*) _mm_malloc(sizeof(double)*size*size, 64);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i < lda && j < lda) {
                copy[i * size + j] = M[i + j * lda];
            } else {
                copy[i * size + j] = 0;
            }
        }
    }
    return copy;
}


double* cache_align_col(double* M, int size) {
    
    double* copy = _mm_malloc(sizeof(double)*size*size, 64);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            copy[i + j * size] = M[i + j * size];
        }
    }
    return copy;
}

void copy_mat(double* C, double* M, int lda, int size) {
    
    for (int i = 0; i < lda; ++i) {
        for (int j = 0; j < lda; ++j) {
                C[i + lda * j] = M[i * size + j];
            }
        }
}
