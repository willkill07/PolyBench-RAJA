/* syrk.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "syrk.hpp"

static void init_array(int n,
                       int m,
                       double *alpha,
                       double *beta,
                       Arr2D<double>* C,
                       Arr2D<double>* A) {
  *alpha = 1.5;
  *beta = 1.2;
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      A->at(i,j) = (double)((i * j + 1) % n) / n;
    }
  );
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      C->at(i,j) = (double)((i * j + 2) % m) / m;
    }
  );
}

static void print_array(int n, const Arr2D<double>* C) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", C->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_syrk(int n,
                        int m,
                        double alpha,
                        double beta,
                        Arr2D<double>* C,
                        Arr2D<double>* A) {
#pragma scop
  RAJA::forallN<Independent2D> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      if (j <= i) {
        C->at(i,j) *= beta;
        RAJA::forall<RAJA::simd_exec> (0, m, [=] (int k) {
          C->at(i,j) += alpha * A->at(i,k) * A->at(j,k);
        });
      }
    }
  );
#pragma endscop
}

int main(int argc, char **argv) {
  int n = N;
  int m = M;
  double alpha;
  double beta;
	Arr2D<double> C { n, n }, A { n, m };

  init_array(n, m, &alpha, &beta, &C, &A);
  {
		util::block_timer t { "SYRK" };
		kernel_syrk(n, m, alpha, beta, &C, &A);
  }
	if (argc > 42)
		print_array(n, &C);
  return 0;
}
