/* syrk.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"

static void init_array(int n,
                       int m,
                       double *alpha,
                       double *beta,
                       double C[1200][1200],
                       double A[1200][1000]) {
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (double)((i * j + 1) % n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (double)((i * j + 2) % m) / m;
}

static void print_array(int n, double C[1200][1200]) {
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

static void kernel_syrk(int n,
                        int m,
                        double alpha,
                        double beta,
                        double C[1200][1200],
                        double A[1200][1000]) {
  int i, j, k;

#pragma scop
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < m; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
#pragma endscop
}

int main(int argc, char **argv) {
  int n = 1200;
  int m = 1000;

  double alpha;
  double beta;
  double(*C)[1200][1200];
  C = (double(*)[1200][1200])polybench_alloc_data((1200) * (1200),
                                                  sizeof(double));
  double(*A)[1200][1000];
  A = (double(*)[1200][1000])polybench_alloc_data((1200) * (1000),
                                                  sizeof(double));

  init_array(n, m, &alpha, &beta, *C, *A);

  polybench_timer_start();

  kernel_syrk(n, m, alpha, beta, *C, *A);

  polybench_timer_stop();
  polybench_timer_print();

  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *C);

  free((void *)C);
  free((void *)A);

  return 0;
}
