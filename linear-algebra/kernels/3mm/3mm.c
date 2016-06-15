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
                       double A[800][1000],
                       double B[1000][900],
                       double C[900][1200],
                       double D[1200][1100]) {
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

static void print_array(int ni, int nl, double G[800][1100]) {
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
                       double E[800][900],
                       double A[800][1000],
                       double B[1000][900],
                       double F[900][1100],
                       double C[900][1200],
                       double D[1200][1100],
                       double G[800][1100]) {
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
  int ni = 800;
  int nj = 900;
  int nk = 1000;
  int nl = 1100;
  int nm = 1200;

  double(*E)[800][900];
  E = (double(*)[800][900])polybench_alloc_data((800) * (900), sizeof(double));
  double(*A)[800][1000];
  A = (double(*)[800][1000])polybench_alloc_data((800) * (1000),
                                                 sizeof(double));
  double(*B)[1000][900];
  B = (double(*)[1000][900])polybench_alloc_data((1000) * (900),
                                                 sizeof(double));
  double(*F)[900][1100];
  F = (double(*)[900][1100])polybench_alloc_data((900) * (1100),
                                                 sizeof(double));
  double(*C)[900][1200];
  C = (double(*)[900][1200])polybench_alloc_data((900) * (1200),
                                                 sizeof(double));
  double(*D)[1200][1100];
  D = (double(*)[1200][1100])polybench_alloc_data((1200) * (1100),
                                                  sizeof(double));
  double(*G)[800][1100];
  G = (double(*)[800][1100])polybench_alloc_data((800) * (1100),
                                                 sizeof(double));

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
