/* mvt.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "mvt.hpp"


static void init_array(int n,
                       double x1[N],
                       double x2[N],
                       double y_1[N],
                       double y_2[N],
                       double A[N][N]) {
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    x1[i] = (double)(i % n) / n;
    x2[i] = (double)((i + 1) % n) / n;
    y_1[i] = (double)((i + 3) % n) / n;
    y_2[i] = (double)((i + 4) % n) / n;
    RAJA::forall<RAJA::simd_exec> (0, n, [=] (int j) {
      A[i][j] = (double)(i * j % n) / n;
    });
  });
}

static void print_array(int n, double x1[N], double x2[N]) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", x1[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "x1");
  fprintf(stderr, "begin dump: %s", "x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", x2[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "x2");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_mvt(int n,
                       double x1[N],
                       double x2[N],
                       double y_1[N],
                       double y_2[N],
                       double A[N][N]) {
#pragma scop
  RAJA::forallN<OuterIndependent2D> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      x1[i] += A[i][j] * y_1[j];
    }
  );
  RAJA::forallN<OuterIndependent2D> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      x2[i] += A[j][i] * y_2[j];
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  double(*A)[N][N];
  A = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  double(*x1)[N];
  x1 = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*x2)[N];
  x2 = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*y_1)[N];
  y_1 = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*y_2)[N];
  y_2 = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  init_array(n, *x1, *x2, *y_1, *y_2, *A);
  polybench_timer_start();
  kernel_mvt(n, *x1, *x2, *y_1, *y_2, *A);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *x1, *x2);
  free((void*)A);
  free((void*)x1);
  free((void*)x2);
  free((void*)y_1);
  free((void*)y_2);
  return 0;
}
