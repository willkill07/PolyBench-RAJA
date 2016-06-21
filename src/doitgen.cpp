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
											 Arr3D<double>* A,
											 Arr2D<double>* C4) {
  RAJA::forallN<Independent3DTiled> (
    RAJA::RangeSegment { 0, nr },
    RAJA::RangeSegment { 0, nq },
    RAJA::RangeSegment { 0, np },
    [=] (int i, int j, int k) {
      A->at(i,j,k) = (double)((i * j + k) % np) / np;
    }
  );
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, np },
    RAJA::RangeSegment { 0, np },
    [=] (int i, int j) {
      C4->at(i,j) = (double)(i * j % np) / np;
    }
  );
}

static void print_array(int nr, int nq, int np, const Arr3D<double>* A) {
  int i, j, k;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
        if ((i * nq * np + j * np + k) % 20 == 0) fprintf(stderr, "\n");
        fprintf(stderr, "%0.2lf ", A->at(i,j,k));
      }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}
void kernel_doitgen(int nr,
                    int nq,
                    int np,
                    Arr3D<double>* A,
                    const Arr2D<double>* C4,
                    Arr3D<double>* sum) {
#pragma scop
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, nr },
    RAJA::RangeSegment { 0, nq },
    [=] (int r, int q) {
      RAJA::forall<RAJA::omp_for_nowait_exec> (0, np, [=] (int p) {
        sum->at(r,q,p) = 0.0;
        RAJA::forall<RAJA::seq_exec> (0, np, [=] (int s) {
          sum->at(r,q,p) += A->at(r,q,s) * C4->at(s,p);
        });
      });
      RAJA::forall<RAJA::simd_exec> (0, np, [=] (int p) {
        A->at(r,q,p) = sum->at(r,q,p);
      });
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int nr = NR;
  int nq = NQ;
  int np = NP;
	Arr3D<double> A { nr, nq, np };
	Arr3D<double> sum { nr, nq, np };
	Arr2D<double> C4 { np, np };

	init_array(nr, nq, np, &A, &C4);
  {
		util::block_timer t { "DOITGEN" };
		kernel_doitgen(nr, nq, np, &A, &C4, &sum);
	}
  if (argc > 42)
		print_array(nr, nq, np, &A);
  return 0;
}
