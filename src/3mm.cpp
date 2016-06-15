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
                       double A[NI][NK],
                       double B[NK][NJ],
                       double C[NJ][NM],
                       double D[NM][NL]) {

  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nk },
    [=] (int i, int j) {
      A[i][j] = (double)((i * j + 1) % ni) / (5 * ni);
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nk },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int j) {
      B[i][j] = (double)((i * (j + 1) + 2) % nj) / (5 * nj);
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nj },
    RAJA::RangeSegment { 0, nm },
    [=] (int i, int j) {
      C[i][j] = (double)(i * (j + 3) % nl) / (5 * nl);
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nm },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int j) {
      D[i][j] = (double)((i * (j + 2) + 2) % nk) / (5 * nk);
    }
  );
}

static void print_array(int ni, int nl, double G[NI][NL]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "G");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", G[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "G");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_3mm(int ni,
                       int nj,
                       int nk,
                       int nl,
                       int nm,
                       double E[NI][NJ],
                       double A[NI][NK],
                       double B[NK][NJ],
                       double F[NJ][NL],
                       double C[NJ][NM],
                       double D[NM][NL],
                       double G[NI][NL]) {

#pragma scop
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nj },
    [=] (int i, int j) {
      double v { 0.0 };
      RAJA::forall <RAJA::simd_exec> (0, nk, [=] (int k) mutable {
        v += A[i][k] * B[k][j];
      });
      E[i][j] = v;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, nj },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int j) {
      double v { 0.0 };
      RAJA::forall <RAJA::simd_exec> (0, nk, [=] (int k) mutable {
        v += C[i][k] * D[k][j];
      });
      F[i][j] = v;
    }
  );
  RAJA::forallN <Independent2DTiled> (
    RAJA::RangeSegment { 0, ni },
    RAJA::RangeSegment { 0, nl },
    [=] (int i, int j) {
      double v { 0.0 };
      RAJA::forall <RAJA::simd_exec> (0, nk, [=] (int k) mutable {
        v += E[i][k] * F[k][j];
      });
      G[i][j] = v;
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
  double(*E)[NI][NJ];
  E = (double(*)[NI][NJ])polybench_alloc_data((NI) * (NJ), sizeof(double));
  double(*A)[NI][NK];
  A = (double(*)[NI][NK])polybench_alloc_data((NI) * (NK), sizeof(double));
  double(*B)[NK][NJ];
  B = (double(*)[NK][NJ])polybench_alloc_data((NK) * (NJ), sizeof(double));
  double(*F)[NJ][NL];
  F = (double(*)[NJ][NL])polybench_alloc_data((NJ) * (NL), sizeof(double));
  double(*C)[NJ][NM];
  C = (double(*)[NJ][NM])polybench_alloc_data((NJ) * (NM), sizeof(double));
  double(*D)[NM][NL];
  D = (double(*)[NM][NL])polybench_alloc_data((NM) * (NL), sizeof(double));
  double(*G)[NI][NL];
  G = (double(*)[NI][NL])polybench_alloc_data((NI) * (NL), sizeof(double));
  init_array(ni, nj, nk, nl, nm, *A, *B, *C, *D);
  polybench_timer_start();
  kernel_3mm(ni, nj, nk, nl, nm, *E, *A, *B, *F, *C, *D, *G);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(ni, nl, *G);
  free((void*)E);
  free((void*)A);
  free((void*)B);
  free((void*)F);
  free((void*)C);
  free((void*)D);
  free((void*)G);
  return 0;
}
