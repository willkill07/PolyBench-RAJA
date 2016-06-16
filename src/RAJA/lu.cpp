/* lu.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "lu.hpp"


static void init_array(int n, double A[N][N]) {
  double(*B)[N][N];
  B = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      A[i][j] = (j < i) ? ((double)(-j % n) / n + 1) : (j == i);
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

static void print_array(int n, double A[N][N]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", A[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_lu(int n, double A[N][N]) {
  int i, j, k;
#pragma scop
  RAJA::forall<RAJA::seq_exec> (0, n, [=] (int i) {
    RAJA::forall<RAJA::omp_parallel_for_exec> (0, i, [=] (int j) {
      RAJA::forall<RAJA::simd_exec> (0, j, [=] (int j) {
        A[i][j] -= A[i][k] * A[k][j];
      });
      A[i][j] /= A[j][j];
    });
    RAJA::forall<RAJA::omp_parallel_for_exec> (i, n, [=] (int j) {
      RAJA::forall<RAJA::simd_exec> (0, i, [=] (int j) {
        A[i][j] -= A[i][k] * A[k][j];
      });
      A[i][j] /= A[j][j];
    });
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  double(*A)[N][N];
  A = (double(*)[N][N])polybench_alloc_data((N) * (N), sizeof(double));
  init_array(n, *A);
  polybench_timer_start();
  kernel_lu(n, *A);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *A);
  free((void*)A);
  return 0;
}
