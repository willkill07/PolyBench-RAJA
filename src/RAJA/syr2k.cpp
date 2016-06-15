/* syr2k.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "syr2k.hpp"


static void init_array(int n,
                       int m,
                       double *alpha,
                       double *beta,
                       double C[N][N],
                       double A[N][M],
                       double B[N][M]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (double)((i * j + 1) % n) / n;
      B[i][j] = (double)((i * j + 2) % m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (double)((i * j + 3) % n) / m;
    }
}

static void print_array(int n, double C[N][N]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", C[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_syr2k(int n,
                         int m,
                         double alpha,
                         double beta,
                         double C[N][N],
                         double A[N][M],
                         double B[N][M]) {
  int i, j, k;
#pragma scop
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < m; k++)
      for (j = 0; j <= i; j++) {
        C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
      }
  }
#pragma endscop
}

int main(int argc, char **argv) {
  int n = N;
  int m = M;
  double alpha;
  double beta;
  double(*C)[N][N];
  C = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  double(*A)[N][M];
  A = (double(*)[N][M])polybench_alloc_data((N) * (M), sizeof(double));
  double(*B)[N][M];
  B = (double(*)[N][M])polybench_alloc_data((N) * (M), sizeof(double));
  init_array(n, m, &alpha, &beta, *C, *A, *B);
  polybench_timer_start();
  kernel_syr2k(n, m, alpha, beta, *C, *A, *B);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *C);
  free((void *)C);
  free((void *)A);
  free((void *)B);
  return 0;
}
