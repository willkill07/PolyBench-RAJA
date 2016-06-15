/* gesummv.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"

static void init_array(int n,
                       double *alpha,
                       double *beta,
                       double A[1300][1300],
                       double B[1300][1300],
                       double x[1300]) {
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

static void print_array(int n, double y[1300])

{
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
                           double A[1300][1300],
                           double B[1300][1300],
                           double tmp[1300],
                           double x[1300],
                           double y[1300]) {
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
  int n = 1300;

  double alpha;
  double beta;
  double(*A)[1300][1300];
  A = (double(*)[1300][1300])polybench_alloc_data((1300) * (1300),
                                                  sizeof(double));
  double(*B)[1300][1300];
  B = (double(*)[1300][1300])polybench_alloc_data((1300) * (1300),
                                                  sizeof(double));
  double(*tmp)[1300];
  tmp = (double(*)[1300])polybench_alloc_data(1300, sizeof(double));
  double(*x)[1300];
  x = (double(*)[1300])polybench_alloc_data(1300, sizeof(double));
  double(*y)[1300];
  y = (double(*)[1300])polybench_alloc_data(1300, sizeof(double));

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
