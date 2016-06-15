/* doitgen.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "doitgen.hpp"


static void init_array(int nr,
                       int nq,
                       int np,
                       double A[NR][NQ][NP],
                       double C4[NP][NP]) {
  int i, j, k;
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
        A[i][j][k] = (double)((i * j + k) % np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (double)(i * j % np) / np;
}

static void print_array(int nr, int nq, int np, double A[NR][NQ][NP]) {
  int i, j, k;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
        if ((i * nq * np + j * np + k) % 20 == 0) fprintf(stderr, "\n");
        fprintf(stderr, "%0.2lf ", A[i][j][k]);
      }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}
void kernel_doitgen(int nr,
                    int nq,
                    int np,
                    double A[NR][NQ][NP],
                    double C4[NP][NP],
                    double sum[NP]) {
  int r, q, p, s;
#pragma scop
  for (r = 0; r < nr; r++)
    for (q = 0; q < nq; q++) {
      for (p = 0; p < np; p++) {
        sum[p] = 0.0;
        for (s = 0; s < np; s++)
          sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < np; p++)
        A[r][q][p] = sum[p];
    }
#pragma endscop
}

int main(int argc, char** argv) {
  int nr = NR;
  int nq = NQ;
  int np = NP;
  double(*A)[NR][NQ][NP];
  A = (double(*)[NR][NQ][NP])polybench_alloc_data((NR) * (NQ) * (NP),
                                                  sizeof(double));
  double(*sum)[NP];
  sum = (double(*)[NP])polybench_alloc_data(NP, sizeof(double));
  double(*C4)[NP][NP];
  C4 = (double(*)[NP][NP])polybench_alloc_data((NP) * (NP), sizeof(double));
  init_array(nr, nq, np, *A, *C4);
  polybench_timer_start();
  kernel_doitgen(nr, nq, np, *A, *C4, *sum);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(nr, nq, np, *A);
  free((void*)A);
  free((void*)sum);
  free((void*)C4);
  return 0;
}
