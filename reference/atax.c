/* atax.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "atax.h"

static void init_array(int m, int n, double A[M][N], double x[N])
{
  int i, j;
  double fn;
  fn = (double)n;
  for (i = 0; i < n; i++)
    x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (double)((i + j) % n) / (5 * m);
}

static void print_array(int n, double y[N])
{
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", y[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "y");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_atax(
  int m,
  int n,
  double A[M][N],
  double x[N],
  double y[N],
  double tmp[M])
{
  int i, j;
#pragma scop
  for (i = 0; i < n; i++)
    y[i] = 0;
  for (i = 0; i < m; i++) {
    tmp[i] = 0.0;
    for (j = 0; j < n; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (j = 0; j < n; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
#pragma endscop
}

int main(int argc, char **argv)
{
  int m = M;
  int n = N;
  double(*A)[M][N];
  A = (double(*)[M][N])polybench_alloc_data((M) * (N), sizeof(double));
  double(*x)[N];
  x = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*y)[N];
  y = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*tmp)[M];
  tmp = (double(*)[M])polybench_alloc_data(M, sizeof(double));
  init_array(m, n, *A, *x);
  polybench_timer_start();
  kernel_atax(m, n, *A, *x, *y, *tmp);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], ""))
    print_array(n, *y);
  free((void *)A);
  free((void *)x);
  free((void *)y);
  free((void *)tmp);
  return 0;
}
