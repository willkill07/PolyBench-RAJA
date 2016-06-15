/* cholesky.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "cholesky.hpp"


static void init_array(int n, double A[N][N]) {
  int i, j;
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
  free((void*)B);
}

static void print_array(int n, double A[N][N]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", A[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_cholesky(int n, double A[N][N]) {
  int i, j, k;
#pragma scop
  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
      for (k = 0; k < j; k++) {
        A[i][j] -= A[i][k] * A[j][k];
      }
      A[i][j] /= A[j][j];
    }
    for (k = 0; k < i; k++) {
      A[i][i] -= A[i][k] * A[i][k];
    }
    A[i][i] = sqrt(A[i][i]);
  }
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  double(*A)[N][N];
  A = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  init_array(n, *A);
  polybench_timer_start();
  kernel_cholesky(n, *A);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *A);
  free((void*)A);
  return 0;
}
