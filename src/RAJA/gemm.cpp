/* gemm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "gemm.hpp"


static void init_array(int ni,
                       int nj,
                       int nk,
                       double *alpha,
                       double *beta,
                       double C[NI][NJ],
                       double A[NI][NK],
                       double B[NK][NJ]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  RAJA::forallN<Independent2D> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int j) {
      C[i][j] = (double)((i * j + 1) % ni) / ni;
    }
  );
  RAJA::forallN<Independent2D> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nk },
    [=] (int i, int j) {
      A[i][j] = (double)(i * (j + 1) % nk) / nk;
    }
  );
  RAJA::forallN<Independent2D> (
    RAJA::RangeSegment { 0, nk },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int j) {
      B[i][j] = (double)(i * (j + 2) % nj) / nj;
    }
  );
}

static void print_array(int ni, int nj, double C[NI][NJ]) {
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
                        double C[NI][NJ],
                        double A[NI][NK],
                        double B[NK][NJ]) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, ni, [=] (int i) {
    RAJA::forall<RAJA::simd_exec> (0, nj, [=] (int j) {
      C[i][j] *= beta;
    });
    RAJA::forallN<
      RAJA::NestedPolicy<
        RAJA::ExecList<RAJA::simd_exec,RAJA::simd_exec>,
        RAJA::Permute<RAJA::PERM_JI>
      >
    >(RAJA::RangeSegment { 0, nk },
      RAJA::RangeSegment { 0, nj },
      [=] (int k, int j) {
        C[i][j] += alpha * A[i][k] * B[k][j];
      });
  });
#pragma endscop
}

int main(int argc, char **argv) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  double alpha;
  double beta;
  double(*C)[NI][NJ];
  C = (double(*)[NI][NJ])polybench_alloc_data((NI) * (NJ), sizeof(double));
  double(*A)[NI][NK];
  A = (double(*)[NI][NK])polybench_alloc_data((NI) * (NK), sizeof(double));
  double(*B)[NK][NJ];
  B = (double(*)[NK][NJ])polybench_alloc_data((NK) * (NJ), sizeof(double));
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
