/* 3mm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "3mm.hpp"

static void init_array(int ni,
                       int nj,
                       int nk,
                       int nl,
                       int nm,
                       Ptr<double> A,
                       Ptr<double> B,
                       Ptr<double> C,
                       Ptr<double> D) {

  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nk },
    [=] (int i, int k) {
      ACC_2D(A,ni,nk,i,k) = (double)((i * k + 1) % ni) / (5 * ni);
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nk },
    RAJA::RangeSegment { 0, nj },
    [=] (int k, int j) {
      ACC_2D(B,nk,nj,k,j) = (double)((k * (j + 1) + 2) % nj) / (5 * nj);
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nj },
    RAJA::RangeSegment { 0, nm },
    [=] (int j, int m) {
      ACC_2D(C,nj,nm,j,m) = (double)(j * (m + 3) % nl) / (5 * nl);
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nm },
    RAJA::RangeSegment { 0, nl },
    [=] (int m, int l) {
      ACC_2D(A,nm,nl,m,l) = (double)((m * (l + 2) + 2) % nk) / (5 * nk);
    }
  );
}

static void print_array(int ni, int nl, const double* __restrict__ G) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", ACC_2D(G,ni,nl,i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "G");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_3mm(int ni,
                       int nj,
                       int nk,
                       int nl,
                       int nm,
                       double* __restrict__ E,
                       const double* __restrict__ A,
                       const double* __restrict__ B,
                       double* __restrict__ F,
                       const double* __restrict__ C,
                       const double* __restrict__ D,
                       double* __restrict__ G) {

#pragma scop
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int j) {
      RAJA::forall <RAJA::simd_exec> (0, nk, [=] (int k) {
        ACC_2D(E,ni,nj,i,j) += ACC_2D(A,ni,nk,i,k) * ACC_2D(B,nk,nj,k,j);
      });
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nj },
    RAJA::RangeSegment { 0, nl },
    [=] (int j, int l) {
      RAJA::forall <RAJA::simd_exec> (0, nk, [=] (int k) {
        ACC_2D(F,nj,nl,j,l) += ACC_2D(C,nj,nk,j,k) * ACC_2D(D,nk,nl,k,l);
      });
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int l) {
      RAJA::forall <RAJA::simd_exec> (0, nj, [=] (int j) {
        ACC_2D(G,ni,nl,i,l) += ACC_2D(E,ni,nj,i,j) * ACC_2D(F,nj,nl,j,l);
      });
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;
  double *E, *A, *B, *F, *C, *D, *G;
  E = (double*)polybench_alloc_data(ni * nj, sizeof(double));
  A = (double*)polybench_alloc_data(ni * nk, sizeof(double));
  B = (double*)polybench_alloc_data(nk * nj, sizeof(double));
  F = (double*)polybench_alloc_data(nj * nl, sizeof(double));
  C = (double*)polybench_alloc_data(nj * nm, sizeof(double));
  D = (double*)polybench_alloc_data(nm * nl, sizeof(double));
  G = (double*)polybench_alloc_data(ni * nl, sizeof(double));
  init_array(ni, nj, nk, nl, nm, A, B, C, D);
  polybench_timer_start();
  kernel_3mm(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(ni, nl, G);

  free (E);
  free (A);
  free (B);
  free (F);
  free (C);
  free (D);
  free (G);

  return 0;
}
