/* syr2k.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "syr2k.hpp"

static void init_array(int n,
                       int m,
                       double *alpha,
                       double *beta,
                       Arr2D<double>* C,
                       Arr2D<double>* A,
                       Arr2D<double>* B) {
  *alpha = 1.5;
  *beta = 1.2;
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      A->at(i,j) = (double)((i * j + 1) % n) / n;
      B->at(i,j) = (double)((i * j + 2) % m) / m;
    }
  );
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      C->at(i,j) = (double)((i * j + 3) % n) / m;
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

static void kernel_syr2k(int n,
                         int m,
                         double alpha,
                         double beta,
                         Arr2D<double>* C,
                         const Arr2D<double>* A,
                         const Arr2D<double>* B) {
#pragma scop
  RAJA::forallN<Independent2D> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      if (j <= i) {
        C->at(i,j) *= beta;
        RAJA::forall<RAJA::simd_exec> (0, m, [=] (int k) {
          C->at(i,j) += A->at(j,k) * alpha * B->at(i,k) + B->at(j,k) * alpha * A->at(i,k);
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
	Arr2D<double> C { n, n }, A { n, m }, B { n, m };

  init_array(n, m, &alpha, &beta, &C, &A, &B);
  {
		util::block_timer t { "SYR2K" };
		kernel_syr2k(n, m, alpha, beta, &C, &A, &B);
	}
	if (argc > 42)
		print_array(n, &C);
  return 0;
}
