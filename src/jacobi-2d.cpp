/* jacobi-2d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "jacobi-2d.hpp"

static void init_array(int n, Arr2D<double>* A, Arr2D<double>* B) {
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      A->at(i,j) = ((double)i * (j + 2) + 2) / n;
      B->at(i,j) = ((double)i * (j + 3) + 3) / n;
    }
  );
}

static void print_array(int n, const Arr2D<double>* A) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", A->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_jacobi_2d(int tsteps,
                             int n,
                             Arr2D<double>* A,
                             Arr2D<double>* B) {
#pragma scop
  RAJA::forall<RAJA::seq_exec> (0, tsteps, [=] (int t) {
    RAJA::forallN<Independent2DTiled> (
      RAJA::RangeSegment { 1, n - 1 },
      RAJA::RangeSegment { 1, n - 1 },
      [=] (int i, int j) {
        B->at(i,j) = 0.2 * (A->at(i,j) + A->at(i,j - 1) + A->at(i,1 + j) + A->at(1 + i,j)
                       + A->at(i - 1,j));
      }
    );
    RAJA::forallN<Independent2DTiled> (
      RAJA::RangeSegment { 1, n - 1 },
      RAJA::RangeSegment { 1, n - 1 },
      [=] (int i, int j) {
        A->at(i,j) = 0.2 * (B->at(i,j) + B->at(i,j - 1) + B->at(i,1 + j) + B->at(1 + i,j)
                       + B->at(i - 1,j));
      }
    );
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int tsteps = TSTEPS;
	Arr2D<double> A { n, n }, B { n, n };

  init_array(n, &A, &B);
	{
		util::block_timer t { "JACOBI-2D" };
		kernel_jacobi_2d(tsteps, n, &A, &B);
	}
	if (argc > 42)
		print_array(n, &A);
  return 0;
}
