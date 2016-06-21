/* trisolv.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "trisolv.hpp"

static void init_array(int n, Arr2D<double>* L, Arr1D<double>* x, Arr1D<double>* b) {
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    x->at(i) = -999;
    b->at(i) = i;
    RAJA::forall<RAJA::simd_exec> (0, i + 1, [=] (int j) {
      L->at(i,j) = (double)(i + n - j + 1) * 2 / n;
    });
  });
}

static void print_array(int n, const Arr1D<double>* x) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x");
  for (i = 0; i < n; i++) {
    fprintf(stderr, "%0.2lf ", x->at(i));
    if (i % 20 == 0) fprintf(stderr, "\n");
  }
  fprintf(stderr, "\nend   dump: %s\n", "x");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_trisolv(int n, const Arr2D<double>* L, Arr1D<double>* x, Arr1D<double>* b) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
		RAJA::ReduceSum<RAJA::seq_reduce, double> v { 0.0 };
		RAJA::forall<RAJA::simd_exec> (0, i, [=] (int j) {
      v += L->at(i,j) * x->at(j);
    });
    x->at(i) = (b->at(i) - v) / L->at(i,i);
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
	Arr2D<double> L { n, n };
	Arr1D<double> x { n }, b { n };

  init_array(n, &L, &x, &b);
	{
		util::block_timer t { "TRISOLV" };
		kernel_trisolv(n, &L, &x, &b);
	}
	if (argc > 42)
		print_array(n, &x);
  return 0;
}
