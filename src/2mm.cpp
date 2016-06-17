/* 2mm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "2mm.hpp"

static void init_array(int ni,
                       int nj,
                       int nk,
                       int nl,
                       double* alpha,
                       double* beta,
                       double* __restrict__ A,
                       double* __restrict__ B,
                       double* __restrict__ C,
                       double* __restrict__ D) {
  *alpha = 1.5;
  *beta = 1.2;

  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nk },
    [=] (int i, int k) {
      ACC_2D(A,ni,nk,i,k) = (double)((i * k + 1) % ni) / ni;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nk },
    RAJA::RangeSegment { 0, nj },
    [=] (int k, int j) {
      ACC_2D(B,nk,nj,k,j) = (double)(k * (j + 1) % nj) / nj;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nj },
    RAJA::RangeSegment { 0, nl },
    [=] (int j, int l) {
      ACC_2D(C,nj,nl,j,l) = (double)((j * (l + 3) + 1) % nl) / nl;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int l) {
      ACC_2D(D,ni,nl,i,l) = (double)(i * (l + 2) % nk) / nk;
    }
  );
}

static void print_array(int ni, int nl, const double* D) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", ACC_2D(D,ni,nl,i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "D");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_2mm(int ni,
                       int nj,
                       int nk,
                       int nl,
                       double alpha,
                       double beta,
                       double* __restrict__ tmp,
                       const double* __restrict__ A,
                       const double* __restrict__ B,
                       const double* __restrict__ C,
                       double* D) {
#pragma scop
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int j) {
      double t { 0.0 };
      RAJA::forall <RAJA::simd_exec> (0, nk, [=,&t] (int k) {
        t += alpha * ACC_2D(A,ni,nk,i,k) * ACC_2D(B,nk,nj,k,j);
      });
      ACC_2D(tmp,ni,nj,i,j) = t;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int l) {
      double t { 0.0 };
      RAJA::forall <RAJA::simd_exec> (0, nj, [=,&t] (int j) {
        t += alpha * ACC_2D(tmp,ni,nj,i,j) * ACC_2D(C,nj,nl,j,l);
      });
      ACC_2D(D,ni,nl,i,l) = t;
    }
  );
#pragma endscop
}

int main(int argc, char **argv) {
  unsigned long ni = NI;
  unsigned long nj = NJ;
  unsigned long nk = NK;
  unsigned long nl = NL;
  double alpha;
  double beta;

  double *A, *B, *C, *D, *tmp;
  A = (double*)polybench_alloc_data(ni * nk, sizeof(double));
  B = (double*)polybench_alloc_data(nk * nj, sizeof(double));
  C = (double*)polybench_alloc_data(nj * nl, sizeof(double));
  D = (double*)polybench_alloc_data(ni * nl, sizeof(double));
  tmp = (double*)polybench_alloc_data(ni * nj, sizeof(double));

  init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);
  polybench_timer_start();
  kernel_2mm(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(ni, nl, D);

  free (tmp);
  free (A);
  free (B);
  free (C);
  free (D);

  return 0;
}
