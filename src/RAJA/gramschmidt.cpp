/* gramschmidt.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "gramschmidt.hpp"


static void init_array(int m,
                       int n,
                       double A[M][N],
                       double R[N][N],
                       double Q[M][N]) {
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((double)((i * j) % m) / m) * 100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}

static void print_array(int m,
                        int n,
                        double A[M][N],
                        double R[N][N],
                        double Q[M][N]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", R[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "R");
  fprintf(stderr, "begin dump: %s", "Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", Q[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "Q");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gramschmidt(int m,
                               int n,
                               double A[M][N],
                               double R[N][N],
                               double Q[M][N]) {
  int i, j, k;
  double nrm;
#pragma scop
  for (k = 0; k < n; k++) {
    nrm = 0.0;
    for (i = 0; i < m; i++)
      nrm += A[i][k] * A[i][k];
    R[k][k] = sqrt(nrm);
    for (i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];
    for (j = k + 1; j < n; j++) {
      R[k][j] = 0.0;
      for (i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (i = 0; i < m; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
#pragma endscop
}

int main(int argc, char** argv) {
  int m = M;
  int n = N;
  double(*A)[M][N];
  A = (double(*)[M][N])polybench_alloc_data((M) * (N), sizeof(double));
  double(*R)[N][N];
  R = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  double(*Q)[M][N];
  Q = (double(*)[M][N])polybench_alloc_data((M) * (N), sizeof(double));
  init_array(m, n, *A, *R, *Q);
  polybench_timer_start();
  kernel_gramschmidt(m, n, *A, *R, *Q);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(m, n, *A, *R, *Q);
  free((void*)A);
  free((void*)R);
  free((void*)Q);
  return 0;
}
