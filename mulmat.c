mulmat.c
#include "mulmat.h"
#include "constantes.h"
#include <omp.h>
#include <string.h> 

// sequencial
void mul_sequencial(double **A, double **B, double **C, int n) {
    for (int i_block = 0; i_block < n; i_block += BLOCK_SIZE) {
        for (int j_block = 0; j_block < n; j_block += BLOCK_SIZE) {

            // bloco pra otimização de cache L1
            double C_bloco[BLOCK_SIZE][BLOCK_SIZE];
            memset(C_bloco, 0, sizeof(C_bloco));

            int i_max = MIN(i_block + BLOCK_SIZE, n);
            int j_max = MIN(j_block + BLOCK_SIZE, n);

            for (int k_block = 0; k_block < n; k_block += BLOCK_SIZE) {
                int k_max = MIN(k_block + BLOCK_SIZE, n);

                for (int i = i_block; i < i_max; ++i) {
                    for (int k = k_block; k < k_max; ++k) {
                        double a_ik = A[i][k];
                        
                        #pragma omp simd
                        for (int j = j_block; j < j_max; ++j) {
                            C_bloco[i - i_block][j - j_block] += a_ik * B[k][j]; 
                        }
                    }
                }
            }

            // copia o resultado do bloco temporário para a matriz principal
            for (int i = i_block; i < i_max; ++i) {
                for (int j = j_block; j < j_max; ++j) {
                    C[i][j] = C_bloco[i - i_block][j - j_block];
                }
            }
        }
    }
}

// 1D
void paralelo_1D(double **A, double **B, double **C, int n) {
    
    // paralelizando o loop de bloco mais externo (no caso i_block)
    #pragma omp parallel for schedule(static)
    for (int i_block = 0; i_block < n; i_block += BLOCK_SIZE) {
        for (int j_block = 0; j_block < n; j_block += BLOCK_SIZE) {

            double C_bloco[BLOCK_SIZE][BLOCK_SIZE];
            memset(C_bloco, 0, sizeof(C_bloco));

            int i_max = MIN(i_block + BLOCK_SIZE, n);
            int j_max = MIN(j_block + BLOCK_SIZE, n);

            for (int k_block = 0; k_block < n; k_block += BLOCK_SIZE) {
                int k_max = MIN(k_block + BLOCK_SIZE, n);
                
                for (int i = i_block; i < i_max; ++i) {
                    for (int k = k_block; k < k_max; ++k) {
                        double a_ik = A[i][k];
                        
                        #pragma omp simd
                        for (int j = j_block; j < j_max; ++j) {
                            C_bloco[i - i_block][j - j_block] += a_ik * B[k][j]; 
                        }
                    }
                }
            }

            // copia o resultado pra
            for (int i = i_block; i < i_max; ++i) {
                for (int j = j_block; j < j_max; ++j) {
                    C[i][j] = C_bloco[i - i_block][j - j_block];
                }
            }
        }
    }
}

// 2D
void paralelo_2D(double **A, double **B, double **C, int n) {

    // paralelizando os loops de bloco i_block e j_block
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i_block = 0; i_block < n; i_block += BLOCK_SIZE) {
        for (int j_block = 0; j_block < n; j_block += BLOCK_SIZE) {

            double C_bloco[BLOCK_SIZE][BLOCK_SIZE];
            memset(C_bloco, 0, sizeof(C_bloco));

            // aqui interando sobre os blocos k
            for (int k_block = 0; k_block < n; k_block += BLOCK_SIZE) {
                
                int i_max = MIN(i_block + BLOCK_SIZE, n);
                int j_max = MIN(j_block + BLOCK_SIZE, n);
                int k_max = MIN(k_block + BLOCK_SIZE, n);

                for (int i = i_block; i < i_max; ++i) {
                    for (int k = k_block; k < k_max; ++k) {
                        double a_ik = A[i][k];
                        
                        #pragma omp simd
                        for (int j = j_block; j < j_max; ++j) {
                            C_bloco[i - i_block][j - j_block] += a_ik * B[k][j];
                        }
                    }
                }
            }
            
            // copia o resultado do bloco temporário pra a matriz principal C
            int i_max = MIN(i_block + BLOCK_SIZE, n);
            int j_max = MIN(j_block + BLOCK_SIZE, n);
            for (int i = i_block; i < i_max; ++i) {
                for (int j = j_block; j < j_max; ++j) {
                    C[i][j] = C_bloco[i - i_block][j - j_block];
                }
            }
        }
    }
}
