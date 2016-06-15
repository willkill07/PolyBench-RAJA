/* 3mm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "3mm.hpp"

RAJA::RangeSegment IDir (0, NI);
RAJA::RangeSegment JDir (0, NJ);
RAJA::RangeSegment KDir (0, NK);
RAJA::RangeSegment LDir (0, NL);
RAJA::RangeSegment MDir (0, NM);


static void init_array(int ni,
                       int nj,
                       int nk,
                       int nl,
                       int nm,
                       double A[NI][NK],
                       double B[NK][NJ],
                       double C[NJ][NM],
                       double D[NM][NL]) {
  int i, j;

  RAJA::forallN <Independent2D> (IDir, KDir, [=] (int i, int j) {
    A[i][j] = (double)((i * j + 1) % ni) / (5 * ni);
  });
  RAJA::forallN <Independent2D> (KDir, JDir, [=] (int i, int j) {
    B[i][j] = (double)((i * (j + 1) + 2) % nj) / (5 * nj);
  });
  RAJA::forallN <Independent2D> (JDir, MDir, [=] (int i, int j) {
    C[i][j] = (double)(i * (j + 3) % nl) / (5 * nl);
  });
  RAJA::forallN <Independent2D> (MDir, LDir, [=] (int i, int j) {
    D[i][j] = (double)((i * (j + 2) + 2) % nk) / (5 * nk);
  });
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
  using ExecPolicy = Independent2DTiled<32,16>;

  RAJA::forallN <ExecPolicy> (IDir, JDir, [=] (int i, int j) {
    E[i][j] = 0.0;
    for (int k = 0; k < nk; ++k)
      E[i][j] += A[i][k] * B[k][j];
  });
  RAJA::forallN <ExecPolicy> (JDir, LDir, [=] (int i, int j) {
    F[i][j] = 0.0;
    for (int k = 0; k < nm; ++k)
      F[i][j] += C[i][k] * D[k][j];
  });
  RAJA::forallN <ExecPolicy> (IDir, LDir, [=] (int i, int j) {
    G[i][j] = 0.0;
    for (int k = 0; k < nj; ++k)
      G[i][j] += E[i][k] * F[k][j];
  });
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
