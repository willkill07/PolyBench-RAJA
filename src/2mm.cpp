
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
                       double *alpha,
                       double *beta,
                       Arr2D<double>* A,
                       Arr2D<double>* B,
                       Arr2D<double>* C,
                       Arr2D<double>* D) {
  *alpha = 1.5;
  *beta = 1.2;

  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nk },
    [=] (int i, int j) {
      A->at(i,j) = (double)((i * j + 1) % ni) / ni;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nk },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int j) {
      B->at(i,j) = (double)(i * (j + 1) % nj) / nj;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nj },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int j) {
      C->at(i,j) = (double)((i * (j + 3) + 1) % nl) / nl;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int j) {
      D->at(i,j) = (double)(i * (j + 2) % nk) / nk;
    }
  );
}

static void print_array(int ni, int nl, const Arr2D<double> D) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", D(i,j));
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
                       Arr2D<double>* tmp,
                       const Arr2D<double>* A,
                       const Arr2D<double>* B,
                       const Arr2D<double>* C,
                       Arr2D<double>* D) {
#pragma scop
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int j) {
      double v { 0.0 };
      RAJA::forall <RAJA::simd_exec> (0, nk, [=] (int k) mutable {
        v += alpha * A->at(i,k) * B->at(k,j);
      });
      tmp->at(i,j) = v;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int j) {
      double v { 0.0 };
      RAJA::forall <RAJA::simd_exec> (0, nk, [=] (int k) mutable {
        v += tmp->at(i,k) * C->at(k,j);
      });
      D->at(i,j) = v;
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

  double *Aptr, *Bptr, *Cptr, *Dptr, *tmpptr;
  Aptr = (double*)polybench_alloc_data(NI * NK, sizeof(double));
  Bptr = (double*)polybench_alloc_data(NK * NJ, sizeof(double));
  Cptr = (double*)polybench_alloc_data(NJ * NL, sizeof(double));
  Dptr = (double*)polybench_alloc_data(NI * NL, sizeof(double));
  tmpptr = (double*)polybench_alloc_data(NI * NJ, sizeof(double));

  Arr2D <double> tmp { { ni, nj }, tmpptr };
  Arr2D <double> A { { ni, nk }, Aptr };
  Arr2D <double> B { { nk, nj }, Bptr };
  Arr2D <double> C { { nj, nl }, Cptr };
  Arr2D <double> D { { ni, nl }, Dptr };

  init_array(ni, nj, nk, nl, &alpha, &beta, &A, &B, &C, &D);
  polybench_timer_start();
  kernel_2mm(ni, nj, nk, nl, alpha, beta, &tmp, &A, &B, &C, &D);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(ni, nl, D);

  free (tmpptr);
  free (Aptr);
  free (Bptr);
  free (Cptr);
  free (Dptr);

  return 0;
}
