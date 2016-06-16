/* atax.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "atax.hpp"

static void init_array(int m, int n, double A[M][N], double x[N]) {
  double fn;
  fn = (double)n;
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    x[i] = 1 + (i / fn);
  });
  RAJA::forallN<Independent2D> (
    RAJA::RangeSegment {0, m},
    RAJA::RangeSegment {0, n},
    [=] (int i, int j) {
    A[i][j] = (double)((i + j) % n) / (5 * m);
  });
}

static void print_array(int n, double y[N]) {
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

static void kernel_atax(int m,
                        int n,
                        double A[M][N],
                        double x[N],
                        double y[N],
                        double tmp[M]) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, std::max(m, n), [=] (int i) {
    if (i < n) y[i] = 0;
    if (i < m) tmp[i] = 0;
  });
  RAJA::forallN<OuterIndependent2D> (
    RAJA::RangeSegment { 0, m },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      tmp[i] += A[i][j] * x[j];
    }
  );
  RAJA::forallN<OuterIndependent2D> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int j, int i) {
      y[j] += A[i][j] * tmp[i];
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int m = M;
  int n = N;
  double(*A)[M][N];
  A = (double(*)[M][N])polybench_alloc_data((M) * (N), sizeof(double));
  double(*x)[N];
  x = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*y)[N];
  y = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*tmp)[M];
  tmp = (double(*)[M])polybench_alloc_data(M, sizeof(double));
  init_array(m, n, *A, *x);
  polybench_timer_start();
  kernel_atax(m, n, *A, *x, *y, *tmp);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *y);
  free((void*)A);
  free((void*)x);
  free((void*)y);
  free((void*)tmp);
  return 0;
}
