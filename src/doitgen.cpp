/* doitgen.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "doitgen.hpp"


static void init_array(int nr,
                       int nq,
                       int np,
                       double A[NR][NQ][NP],
                       double C4[NP][NP]) {
  RAJA::forallN<Independent3DTiled> (
    RAJA::RangeSegment { 0, nr },
    RAJA::RangeSegment { 0, nq },
    RAJA::RangeSegment { 0, np },
    [=] (int i, int j, int k) {
      A[i][j][k] = (double)((i * j + k) % np) / np;
    }
  );
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, np },
    RAJA::RangeSegment { 0, np },
    [=] (int i, int j) {
      C4[i][j] = (double)(i * j % np) / np;
    }
  );
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
                    double sum[NR][NQ][NP]) {
  int r, q, p, s;
#pragma scop
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, nr },
    RAJA::RangeSegment { 0, nq },
    [=] (int r, int q) {
      RAJA::forall<RAJA::omp_for_nowait_exec> (0, np, [=] (int p) {
        sum[r][q][p] = 0.0;
        RAJA::forall<RAJA::seq_exec> (0, np, [=] (int s) {
          sum[r][q][p] += A[r][q][s] * C4[s][p];
        });
      });
      RAJA::forall<RAJA::simd_exec> (0, np, [=] (int p) {
        A[r][q][p] = sum[r][q][p];
      });
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int nr = NR;
  int nq = NQ;
  int np = NP;
  double(*A)[NR][NQ][NP];
  A = (double(*)[NR][NQ][NP])polybench_alloc_data((NR) * (NQ) * (NP),
                                                  sizeof(double));
  double(*sum)[NR][NQ][NP];
  sum = (double(*)[NR][NQ][NP])polybench_alloc_data((NR) * (NQ) * (NP),
                                                    sizeof(double));
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
