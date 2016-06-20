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
                       Arr2D<double>* A,
                       Arr2D<double>* B,
                       Arr2D<double>* C,
                       Arr2D<double>* D) {
  *alpha = 1.5;
  *beta = 1.2;

  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nk },
    [=] (int i, int k) {
      A->at(i,k) = (double)((i * k + 1) % ni) / ni;
    }
  );

  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nk },
    RAJA::RangeSegment { 0, nj },
    [=] (int k, int j) {
      B->at(k,j) = (double)(k * (j + 1) % nj) / nj;
    }
  );

  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nj },
    RAJA::RangeSegment { 0, nl },
    [=] (int j, int l) {
      C->at(j,l) = (double)((j * (l + 3) + 1) % nl) / nl;
    }
  );

  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int l) {
      D->at(i,l) = (double)(i * (l + 2) % nk) / nk;
    }
  );
}

static void print_array(int ni, int nl, const Arr2D<double>* D) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "D");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", D->at(i,j));
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
  using ExecPolicy = RAJA::NestedPolicy<
    RAJA::ExecList<
      RAJA::omp_collapse_nowait_exec,RAJA::omp_collapse_nowait_exec,RAJA::simd_exec
    >, RAJA::Tile<
      RAJA::TileList<RAJA::tile_fixed<16>,RAJA::tile_fixed<16>,RAJA::tile_none
    >, RAJA::OMP_Parallel<RAJA::Permute<RAJA::PERM_IJK> > > >;

  RAJA::forallN <ExecPolicy> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nj },
    RAJA::RangeSegment { 0, nk },
    [=] (int i, int j, int k) {
      tmp->at(i,j) += alpha * A->at(i,k) * B->at(k,j);
    }
  );
  RAJA::forallN <ExecPolicy> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nl },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int l, int j) {
      D->at(i,l) += alpha * tmp->at(i,j) * C->at(j,l);
    }
  );
#pragma endscop
}

int main(int argc, char **argv) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  double alpha;
  double beta;

  Arr2D<double> A { ni, nk };
  Arr2D<double> B { nk, nj };
  Arr2D<double> C { nj, nl };
  Arr2D<double> D { ni, nl };
  Arr2D<double> tmp { ni, nj };

  init_array(ni, nj, nk, nl, &alpha, &beta, &A, &B, &C, &D);
  polybench_timer_start();
  kernel_2mm(ni, nj, nk, nl, alpha, beta, &tmp, &A, &B, &C, &D);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(ni, nl, &D);
  return 0;
}
