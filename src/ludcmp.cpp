/* ludcmp.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "ludcmp.hpp"


static void init_array(int n,
                       double A[N][N],
                       double b[N],
                       double x[N],
                       double y[N]) {
  double fn = (double)n;
  double(*B)[N][N];
  B = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    x[i] = 0;
    y[i] = 0;
    b[i] = (i + 1) / fn / 2.0 + 4;
  });
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      A[i][j] = (j < i) ? ((double)(-j % n) / n + 1) : (i == j);
    }
  );
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int r, int s) {
      (*B)[r][s] = 0;
    }
  );
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int t, int r) {
      RAJA::forall<RAJA::simd_exec> (0, n, [=] (int s) {
        (*B)[r][s] += A[r][t] * A[s][t];
      });
    }
  );
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int r, int s) {
      A[r][s] = (*B)[r][s];
    }
  );
  free((void*)B);
}

static void print_array(int n, double x[N]) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", x[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "x");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_ludcmp(int n,
                          double A[N][N],
                          double b[N],
                          double x[N],
                          double y[N]) {
#pragma scop
  RAJA::forall<RAJA::seq_exec> (0, n, [=] (int i) {
    RAJA::forall<RAJA::omp_parallel_for_exec> (0, i, [=] (int j) {
      RAJA::forall<RAJA::simd_exec> (0, j, [=] (int k) {
        A[i][j] -= A[i][k] * A[k][j];
      });
      A[i][j] /= A[j][j];
    });
    RAJA::forall<RAJA::omp_parallel_for_exec> (i, n, [=] (int j) {
      RAJA::forall<RAJA::simd_exec> (0, i, [=] (int k) {
        A[i][j] -= A[i][k] * A[k][j];
      });
    });
  });
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    RAJA::forall<RAJA::simd_exec> (0, i, [=] (int j) {
      b[i] -= A[i][j] * y[j];
    });
    y[i] = b[i];
  });
  RAJA::forall<RAJA::omp_parallel_for_exec> (n - 1, -1, -1, [=] (int i) {
    RAJA::forall<RAJA::simd_exec> (0, i, [=] (int j) {
      y[i] -= A[i][j] * x[j];
    });
    x[i] = y[i] / A[i][i];
  });
#pragma endscop
}

int main(int argc, char** argv) {
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
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *x);
  free((void*)A);
  free((void*)b);
  free((void*)x);
  free((void*)y);
  return 0;
}
