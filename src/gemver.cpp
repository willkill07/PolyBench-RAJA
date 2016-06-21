/* gemver.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "gemver.hpp"

static void init_array(int n,
                       double *alpha,
                       double *beta,
                       Arr2D<double>* A,
                       Arr1D<double>* u1,
                       Arr1D<double>* v1,
                       Arr1D<double>* u2,
                       Arr1D<double>* v2,
                       Arr1D<double>* w,
                       Arr1D<double>* x,
                       Arr1D<double>* y,
                       Arr1D<double>* z) {
  *alpha = 1.5;
  *beta = 1.2;
  double fn = (double)n;
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    u1->at(i) = i;
    u2->at(i) = ((i + 1) / fn) / 2.0;
    v1->at(i) = ((i + 1) / fn) / 4.0;
    v2->at(i) = ((i + 1) / fn) / 6.0;
    y->at(i) = ((i + 1) / fn) / 8.0;
    z->at(i) = ((i + 1) / fn) / 9.0;
    x->at(i) = 0.0;
    w->at(i) = 0.0;
    RAJA::forall<RAJA::simd_exec> (0, n, [=] (int j) {
      A->at(i,j) = (double)(i * j % n) / n;
    });
  });
}

static void print_array(int n, const Arr1D<double>* w) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", w->at(i));
  }
  fprintf(stderr, "\nend   dump: %s\n", "w");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gemver(int n,
                          double alpha,
                          double beta,
                          Arr2D<double>* A,
                          const Arr1D<double>* u1,
                          const Arr1D<double>* v1,
                          const Arr1D<double>* u2,
                          const Arr1D<double>* v2,
                          Arr1D<double>* w,
                          Arr1D<double>* x,
                          const Arr1D<double>* y,
                          const Arr1D<double>* z) {
#pragma scop
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      A->at(i,j) = A->at(i,j) + u1->at(i) * v1->at(j) + u2->at(i) * v2->at(j);
    }
  );
  RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::simd_exec>>> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      x->at(i) = x->at(i) + beta * A->at(j,i) * y->at(j);
    }
  );
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    x->at(i) = x->at(i) + z->at(i);
  });
  RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,RAJA::simd_exec>>> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      w->at(i) = w->at(i) + alpha * A->at(i,j) * x->at(j);
    }
  );
#pragma endscop
}

int main(int argc, char **argv) {
  int n = N;
  double alpha;
  double beta;
	Arr2D<double> A { n, n };
	Arr1D<double> u1 { n }, w { n };
	Arr1D<double> v1 { n }, x { n };
	Arr1D<double> u2 { n }, y { n };
	Arr1D<double> v2 { n }, z { n };

	init_array(n, &alpha, &beta, &A, &u1, &v1, &u2, &v2, &w, &x, &y, &z);
	{
		util::block_timer t { "GEMVER" };
		kernel_gemver(n, alpha, beta, &A, &u1, &v1, &u2, &v2, &w, &x, &y, &z);
  }
  if (argc > 42)
		print_array(n, &w);
  return 0;
}
