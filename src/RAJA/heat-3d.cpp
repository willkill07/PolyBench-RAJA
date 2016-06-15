/* heat-3d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "heat-3d.hpp"


static void init_array(int n, double A[N][N][N], double B[N][N][N]) {
  int i, j, k;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (double)(i + j + (n - k)) * 10 / (n);
}

static void print_array(int n, double A[N][N][N]) {
  int i, j, k;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
        if ((i * n * n + j * n + k) % 20 == 0) fprintf(stderr, "\n");
        fprintf(stderr, "%0.2lf ", A[i][j][k]);
      }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_heat_3d(int tsteps,
                           int n,
                           double A[N][N][N],
                           double B[N][N][N]) {
  int t, i, j, k;
#pragma scop
  for (t = 1; t <= TSTEPS; t++) {
    for (i = 1; i < n - 1; i++) {
      for (j = 1; j < n - 1; j++) {
        for (k = 1; k < n - 1; k++) {
          B[i][j][k] =
              0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k])
              + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k])
              + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1])
              + A[i][j][k];
        }
      }
    }
    for (i = 1; i < n - 1; i++) {
      for (j = 1; j < n - 1; j++) {
        for (k = 1; k < n - 1; k++) {
          A[i][j][k] =
              0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k])
              + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k])
              + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1])
              + B[i][j][k];
        }
      }
    }
  }
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int tsteps = TSTEPS;
  double(*A)[N][N][N];
  A = (double(*)[N][N][N])polybench_alloc_data((N) * (N) * (N), sizeof(double));
  double(*B)[N][N][N];
  B = (double(*)[N][N][N])polybench_alloc_data((N) * (N) * (N), sizeof(double));
  init_array(n, *A, *B);
  polybench_timer_start();
  kernel_heat_3d(tsteps, n, *A, *B);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *A);
  free((void*)A);
  return 0;
}
