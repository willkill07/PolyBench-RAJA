/* 3mm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "3mm.h"

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
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double)((i * j + 1) % ni) / (5 * ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double)((i * (j + 1) + 2) % nj) / (5 * nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (double)(i * (j + 3) % nl) / (5 * nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (double)((i * (j + 2) + 2) % nk) / (5 * nk);
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
  int i, j, k;
#pragma scop
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      E[i][j] = 0.0;
      for (k = 0; k < nk; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++) {
      F[i][j] = 0.0;
      for (k = 0; k < nm; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
      G[i][j] = 0.0;
      for (k = 0; k < nj; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
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
