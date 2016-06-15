/* trisolv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"

static void init_array(int n,
                       double L[2000][2000],
                       double x[2000],
                       double b[2000]) {
  int i, j;

  for (i = 0; i < n; i++) {
    x[i] = -999;
    b[i] = i;
    for (j = 0; j <= i; j++)
      L[i][j] = (double)(i + n - j + 1) * 2 / n;
  }
}

static void print_array(int n, double x[2000])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x");
  for (i = 0; i < n; i++) {
    fprintf(stderr, "%0.2lf ", x[i]);
    if (i % 20 == 0) fprintf(stderr, "\n");
  }
  fprintf(stderr, "\nend   dump: %s\n", "x");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_trisolv(int n,
                           double L[2000][2000],
                           double x[2000],
                           double b[2000]) {
  int i, j;

#pragma scop
  for (i = 0; i < n; i++) {
    x[i] = b[i];
    for (j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
#pragma endscop
}

int main(int argc, char** argv) {
  int n = 2000;

  double(*L)[2000][2000];
  L = (double(*)[2000][2000])polybench_alloc_data((2000) * (2000),
                                                  sizeof(double));
  double(*x)[2000];
  x = (double(*)[2000])polybench_alloc_data(2000, sizeof(double));
  double(*b)[2000];
  b = (double(*)[2000])polybench_alloc_data(2000, sizeof(double));

  init_array(n, *L, *x, *b);

  polybench_timer_start();

  kernel_trisolv(n, *L, *x, *b);

  polybench_timer_stop();
  polybench_timer_print();

  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *x);

  free((void*)L);
  free((void*)x);
  free((void*)b);

  return 0;
}
