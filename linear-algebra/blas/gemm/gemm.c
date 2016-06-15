/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"

static void init_array(int ni,
                       int nj,
                       int nk,
                       double *alpha,
                       double *beta,
                       double C[1000][1100],
                       double A[1000][1200],
                       double B[1200][1100]) {
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (double)((i * j + 1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double)(i * (j + 1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double)(i * (j + 2) % nj) / nj;
}

static void print_array(int ni, int nj, double C[1000][1100]) {
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", C[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gemm(int ni,
                        int nj,
                        int nk,
                        double alpha,
                        double beta,
                        double C[1000][1100],
                        double A[1000][1200],
                        double B[1200][1100]) {
  int i, j, k;
# 71 "gemm.c"
#pragma scop
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (k = 0; k < nk; k++) {
      for (j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop
}

int main(int argc, char **argv) {
  int ni = 1000;
  int nj = 1100;
  int nk = 1200;

  double alpha;
  double beta;
  double(*C)[1000][1100];
  C = (double(*)[1000][1100])polybench_alloc_data((1000) * (1100),
                                                  sizeof(double));
  double(*A)[1000][1200];
  A = (double(*)[1000][1200])polybench_alloc_data((1000) * (1200),
                                                  sizeof(double));
  double(*B)[1200][1100];
  B = (double(*)[1200][1100])polybench_alloc_data((1200) * (1100),
                                                  sizeof(double));

  init_array(ni, nj, nk, &alpha, &beta, *C, *A, *B);

  polybench_timer_start();

  kernel_gemm(ni, nj, nk, alpha, beta, *C, *A, *B);

  polybench_timer_stop();
  polybench_timer_print();

  if (argc > 42 && !strcmp(argv[0], "")) print_array(ni, nj, *C);

  free((void *)C);
  free((void *)A);
  free((void *)B);

  return 0;
}
