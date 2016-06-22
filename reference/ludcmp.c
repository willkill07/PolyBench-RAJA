/* ludcmp.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "ludcmp.h"

static void init_array(
  int n,
  double A[N][N],
  double b[N],
  double x[N],
  double y[N]) {
  int i, j;
  double fn = (double)n;
  for (i = 0; i < n; i++) {
    x[i] = 0;
    y[i] = 0;
    b[i] = (i + 1) / fn / 2.0 + 4;
  }
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      A[i][j] = (double)(-j % n) / n + 1;
    for (j = i + 1; j < n; j++) {
      A[i][j] = 0;
    }
    A[i][i] = 1;
  }
  int r, s, t;
  double(*B)[N][N];
  B = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (*B)[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
        (*B)[r][s] += A[r][t] * A[s][t];
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      A[r][s] = (*B)[r][s];
  free((void *)B);
}

static void print_array(int n, double x[N]) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", x[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "x");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_ludcmp(
  int n,
  double A[N][N],
  double b[N],
  double x[N],
  double y[N]) {
  int i, j, k;
  double w;
#pragma scop
  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
      w = A[i][j];
      for (k = 0; k < j; k++) {
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w / A[j][j];
    }
    for (j = i; j < n; j++) {
      w = A[i][j];
      for (k = 0; k < i; k++) {
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w;
    }
  }
  for (i = 0; i < n; i++) {
    w = b[i];
    for (j = 0; j < i; j++)
      w -= A[i][j] * y[j];
    y[i] = w;
  }
  for (i = n - 1; i >= 0; i--) {
    w = y[i];
    for (j = i + 1; j < n; j++)
      w -= A[i][j] * x[j];
    x[i] = w / A[i][i];
  }
#pragma endscop
}

int main(int argc, char **argv) {
  int n = N;
  double(*A)[N][N];
  A = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  double(*b)[N];
  b = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*x)[N];
  x = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*y)[N];
  y = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  init_array(n, *A, *b, *x, *y);
  polybench_timer_start();
  kernel_ludcmp(n, *A, *b, *x, *y);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], ""))
    print_array(n, *x);
  free((void *)A);
  free((void *)b);
  free((void *)x);
  free((void *)y);
  return 0;
}
