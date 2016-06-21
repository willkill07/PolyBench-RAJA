/* durbin.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "durbin.hpp"

static void init_array(int n, Arr2D<double>* r) {
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
		r->at(i) = (n + 1 - i);
  });
}

static void print_array(int n, double y[N]) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", y[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "y");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_durbin(int n, const Arr1D<double>* r, Arr1D<double>* y) {
	Arr1D<double> _z { n }, *z { &_z };
	double _alpha, _beta, *alpha { &_alpha }, *beta { &_beta };
#pragma scop
  y->at(0) = -r->at(0);
  *beta = 1.0;
  *alpha = -r->at(0);
  RAJA::forall<RAJA::seq_exec> (1, n, [=] (int k) {
    *beta = (1 - *alpha * *alpha) * *beta;
		RAJA::ReduceSum<RAJA::omp_reduce, double> sum { 0.0 };
    RAJA::forall<RAJA::omp_parallel_for_exec> (0, k, [=] (int i) {
      sum += r->at(k - i - 1) * y->at(i);
    });
    alpha = -(r->at(k) + sum) / beta;
    RAJA::forall<RAJA::omp_parallel_for_exec> (0, k, [=] (int i) {
      z->at(i) = y->at(i) + *alpha * y->at(k - i - 1);
    });
    RAJA::forall<RAJA::omp_parallel_for_exec> (0, k, [=] (int i) {
      y->at(i) = z->at(i);
    });
    y->at(k) = *alpha;
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
	Arr1D<double> r { n }, y { n };
  init_array(n, &r);
  {
		util::block_timer t { "DURBIN" }
		kernel_durbin(n, &r, &y);
	}
	if (argc > 42)
		print_array(n, &y);
  return 0;
}
