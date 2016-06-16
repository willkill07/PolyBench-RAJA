/* symm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "symm.hpp"


static void init_array(int m,
                       int n,
                       double *alpha,
                       double *beta,
                       double C[M][N],
                       double A[M][M],
                       double B[M][N]) {
  *alpha = 1.5;
  *beta = 1.2;
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, m },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      C[i][j] = (double)((i + j) % 100) / m;
      B[i][j] = (double)((n + i - j) % 100) / m;
      A[i][j] = (j <= i) ? ((double)((i + j) % 100) / m) : (-999);
    }
  );
}

static void print_array(int m, int n, double C[M][N]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", C[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_symm(int m,
                        int n,
                        double alpha,
                        double beta,
                        double C[M][N],
                        double A[M][M],
                        double B[M][N]) {
#pragma scop
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, m },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      double temp2 = 0;
      RAJA::forall<RAJA::simd_exec> (0, i, [=] (int k) {
        // TODO - Fix update to C
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      });
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
  );
#pragma endscop
}

int main(int argc, char **argv) {
  int m = M;
  int n = N;
  double alpha;
  double beta;
  double(*C)[M][N];
  C = (double(*)[M][N])polybench_alloc_data((M) * (N), sizeof(double));
  double(*A)[M][M];
  A = (double(*)[M][M])polybench_alloc_data((M) * (M), sizeof(double));
  double(*B)[M][N];
  B = (double(*)[M][N])polybench_alloc_data((M) * (N), sizeof(double));
  init_array(m, n, &alpha, &beta, *C, *A, *B);
  polybench_timer_start();
  kernel_symm(m, n, alpha, beta, *C, *A, *B);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(m, n, *C);
  free((void *)C);
  free((void *)A);
  free((void *)B);
  return 0;
}
