/* gesummv.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "gesummv.hpp"


static void init_array(int n,
                       double *alpha,
                       double *beta,
                       double A[N][N],
                       double B[N][N],
                       double x[N]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++) {
    x[i] = (double)(i % n) / n;
    for (j = 0; j < n; j++) {
      A[i][j] = (double)((i * j + 1) % n) / n;
      B[i][j] = (double)((i * j + 2) % n) / n;
    }
  }
}

static void print_array(int n, double y[N]) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", y[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "y");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gesummv(int n,
                           double alpha,
                           double beta,
                           double A[N][N],
                           double B[N][N],
                           double tmp[N],
                           double x[N],
                           double y[N]) {
  int i, j;
#pragma scop
  for (i = 0; i < n; i++) {
    tmp[i] = 0.0;
    y[i] = 0.0;
    for (j = 0; j < n; j++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
#pragma endscop
}

int main(int argc, char **argv) {
  int n = N;
  double alpha;
  double beta;
  double(*A)[N][N];
  A = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  double(*B)[N][N];
  B = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  double(*tmp)[N];
  tmp = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*x)[N];
  x = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*y)[N];
  y = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  init_array(n, &alpha, &beta, *A, *B, *x);
  polybench_timer_start();
  kernel_gesummv(n, alpha, beta, *A, *B, *tmp, *x, *y);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *y);
  free((void *)A);
  free((void *)B);
  free((void *)tmp);
  free((void *)x);
  free((void *)y);
  return 0;
}
