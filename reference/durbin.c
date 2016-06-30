/* durbin.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "durbin.h"

static void init_array(int n, double r[N])
{
  int i, j;
  for (i = 0; i < n; i++) {
    r[i] = (n + 1 - i);
  }
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

static void kernel_durbin(int n, double r[N], double y[N])
{
  double z[N];
  double alpha;
  double beta;
  double sum;
  int i, k;
#pragma scop
  y[0] = -r[0];
  beta = 1.0;
  alpha = -r[0];
  for (k = 1; k < n; k++) {
    beta = (1 - alpha * alpha) * beta;
    sum = 0.0;
    for (i = 0; i < k; i++) {
      sum += r[k - i - 1] * y[i];
    }
    alpha = -(r[k] + sum) / beta;
    for (i = 0; i < k; i++) {
      z[i] = y[i] + alpha * y[k - i - 1];
    }
    for (i = 0; i < k; i++) {
      y[i] = z[i];
    }
    y[k] = alpha;
  }
#pragma endscop
}

int main(int argc, char **argv)
{
  int n = N;
  double(*r)[N];
  r = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*y)[N];
  y = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  init_array(n, *r);
  polybench_timer_start();
  kernel_durbin(n, *r, *y);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], ""))
    print_array(n, *y);
  free((void *)r);
  free((void *)y);
  return 0;
}
