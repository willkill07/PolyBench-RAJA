/* jacobi-1d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "jacobi-1d.hpp"

static void init_array(int n, Arr1D<double>* A, Arr1D<double>* B) {
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    A->at(i) = ((double)i + 2) / n;
    B->at(i) = ((double)i + 3) / n;
  });
}

static void print_array(int n, const Arr1D<double>* A) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", A->at(i));
  }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_jacobi_1d(int tsteps, int n, Arr1D<double>* A, Arr1D<double>* B) {
#pragma scop
  RAJA::forall<RAJA::seq_exec> (0, tsteps, [=] (int t) {
    RAJA::forall<RAJA::omp_parallel_for_exec> (1, n - 1, [=] (int i) {
      B->at(i) = 0.33333 * (A->at(i - 1) + A->at(i) + A->at(i + 1));
    });
    RAJA::forall<RAJA::omp_parallel_for_exec> (1, n - 1, [=] (int i) {
      A->at(i) = 0.33333 * (B->at(i - 1) + B->at(i) + B->at(i + 1));
    });
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int tsteps = TSTEPS;
	Arr1D<double> A { n }, B { n };

  init_array(n, &A, &B);
  {
		util::block_timer t { "JACOBI-1D" };
		kernel_jacobi_1d(tsteps, n, &A, &B);
	}
	if (argc > 42)
		print_array(n, &A);
  return 0;
}
