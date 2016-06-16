/* jacobi-1d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "jacobi-1d.hpp"


static void init_array(int n, double A[N], double B[N]) {
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    A[i] = ((double)i + 2) / n;
    B[i] = ((double)i + 3) / n;
  });
}

static void print_array(int n, double A[N]) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", A[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_jacobi_1d(int tsteps, int n, double A[N], double B[N]) {
#pragma scop
  RAJA::forall<RAJA::seq_exec> (0, tsteps, [=] (int t) {
    RAJA::forall<RAJA::omp_parallel_for_exec> (1, n - 1, [=] (int i) {
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
    });
    RAJA::forall<RAJA::omp_parallel_for_exec> (1, n - 1, [=] (int i) {
      A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1]);
    });
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int tsteps = TSTEPS;
  double(*A)[N];
  A = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*B)[N];
  B = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  init_array(n, *A, *B);
  polybench_timer_start();
  kernel_jacobi_1d(tsteps, n, *A, *B);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *A);
  free((void*)A);
  free((void*)B);
  return 0;
}