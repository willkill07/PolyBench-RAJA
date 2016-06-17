/* gesummv.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "gesummv.hpp"

static void init_array(int n,
                       double* alpha,
                       double* beta,
                       Arr2D<double>* A,
                       Arr2D<double>* B,
                       Arr1D<double>* x) {
  *alpha = 1.5;
  *beta = 1.2;
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, n, [=](int i) {
    x->at(i) = (double)(i % n) / n;
    RAJA::forall<RAJA::simd_exec>(0, n, [=](int j) {
      A->at(i, j) = (double)((i * j + 1) % n) / n;
      B->at(i, j) = (double)((i * j + 2) % n) / n;
    });
  });
}

static void print_array(int n, const Arr1D<double>* y) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", y->at(i));
  }
  fprintf(stderr, "\nend   dump: %s\n", "y");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gesummv(int n,
                           double alpha,
                           double beta,
                           const Arr2D<double>* A,
                           const Arr2D<double>* B,
                           Arr1D<double>* tmp,
                           const Arr1D<double>* x,
                           Arr1D<double>* y) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, n, [=](int i) {
    tmp->at(i) = 0.0;
    y->at(i) = 0.0;
    RAJA::forall<RAJA::simd_exec>(0, n, [=](int j) {
      tmp->at(i) += A->at(i, j) * x->at(j);
      y->at(i) += B->at(i, j) * x->at(j);
    });
    y->at(i) = alpha * tmp->at(i) + beta * y->at(i);
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  double alpha;
  double beta;
  Arr2D<double> A{n, n}, B{n, n};
  Arr1D<double> tmp{n}, x{n}, y{n};

  init_array(n, &alpha, &beta, &A, &B, &x);
  {
    util::block_timer t{"GESUMMV"};
    kernel_gesummv(n, alpha, beta, &A, &B, &tmp, &x, &y);
  }
  if (argc > 42) print_array(n, &y);
  return 0;
}
