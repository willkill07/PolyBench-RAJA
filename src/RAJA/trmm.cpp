/* trmm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "trmm.hpp"


static void init_array(int m,
                       int n,
                       double* alpha,
                       double A[M][M],
                       double B[M][N]) {
  int i, j;
  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (double)((i + j) % m) / m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (double)((n + (i - j)) % n) / n;
    }
  }
}

static void print_array(int m, int n, double B[M][N]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", B[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "B");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_trmm(int m,
                        int n,
                        double alpha,
                        double A[M][M],
                        double B[M][N]) {
  int i, j, k;
# 68 "trmm.c"
#pragma scop
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      for (k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
#pragma endscop
}

int main(int argc, char** argv) {
  int m = M;
  int n = N;
  double alpha;
  double(*A)[M][M];
  A = (double(*)[M][M])polybench_alloc_data((M) * (M), sizeof(double));
  double(*B)[M][N];
  B = (double(*)[M][N])polybench_alloc_data((M) * (N), sizeof(double));
  init_array(m, n, &alpha, *A, *B);
  polybench_timer_start();
  kernel_trmm(m, n, alpha, *A, *B);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(m, n, *B);
  free((void*)A);
  free((void*)B);
  return 0;
}
