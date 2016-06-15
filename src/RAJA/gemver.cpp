/* gemver.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "gemver.hpp"


static void init_array(int n,
                       double *alpha,
                       double *beta,
                       double A[N][N],
                       double u1[N],
                       double v1[N],
                       double u2[N],
                       double v2[N],
                       double w[N],
                       double x[N],
                       double y[N],
                       double z[N]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  double fn = (double)n;
  for (i = 0; i < n; i++) {
    u1[i] = i;
    u2[i] = ((i + 1) / fn) / 2.0;
    v1[i] = ((i + 1) / fn) / 4.0;
    v2[i] = ((i + 1) / fn) / 6.0;
    y[i] = ((i + 1) / fn) / 8.0;
    z[i] = ((i + 1) / fn) / 9.0;
    x[i] = 0.0;
    w[i] = 0.0;
    for (j = 0; j < n; j++)
      A[i][j] = (double)(i * j % n) / n;
  }
}

static void print_array(int n, double w[N]) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", w[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "w");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gemver(int n,
                          double alpha,
                          double beta,
                          double A[N][N],
                          double u1[N],
                          double v1[N],
                          double u2[N],
                          double v2[N],
                          double w[N],
                          double x[N],
                          double y[N],
                          double z[N]) {
  int i, j;
#pragma scop
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];
  for (i = 0; i < n; i++)
    x[i] = x[i] + z[i];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      w[i] = w[i] + alpha * A[i][j] * x[j];
#pragma endscop
}

int main(int argc, char **argv) {
  int n = N;
  double alpha;
  double beta;
  double(*A)[N][N];
  A = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  double(*u1)[N];
  u1 = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*v1)[N];
  v1 = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*u2)[N];
  u2 = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*v2)[N];
  v2 = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*w)[N];
  w = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*x)[N];
  x = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*y)[N];
  y = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*z)[N];
  z = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  init_array(n, &alpha, &beta, *A, *u1, *v1, *u2, *v2, *w, *x, *y, *z);
  polybench_timer_start();
  kernel_gemver(n, alpha, beta, *A, *u1, *v1, *u2, *v2, *w, *x, *y, *z);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *w);
  free((void *)A);
  free((void *)u1);
  free((void *)v1);
  free((void *)u2);
  free((void *)v2);
  free((void *)w);
  free((void *)x);
  free((void *)y);
  free((void *)z);
  return 0;
}
