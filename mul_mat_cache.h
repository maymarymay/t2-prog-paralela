#ifndef MUL_MAT_CACHE_OPT_H
#define MUL_MAT_CACHE_OPT_H

// funções para multiplicação de matrizes
void mul_sequencial(double **A, double **B, double **C, int n);
void paralelo_1D(double **A, double **B, double **C, int n);
void paralelo_2D(double **A, double **B, double **C, int n);

#endif
