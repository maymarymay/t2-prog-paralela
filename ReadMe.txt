# como executar
> export OMP_NUM_THREADS=8
> gcc main.c mul_mat_cache.c -o matmul -fopenmp (lento)
ou
> gcc main.c mul_mat_cache.c -o matmul -O3 -fopenmp -march=native (mais rapido)
> ./matmul 512
> ./matmul 1024
> ./matmul 2048
> ./matmul 4096
