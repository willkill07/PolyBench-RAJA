/* atax.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "atax.hpp"

static void init_array(int m, int n, Arr2D<double>* A, Arr1D<double>* x) {
  double fn;
  fn = (double)n;
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, n, [=](int i) {
    x->at(i) = 1 + (i / fn);
  });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, m},
                               RAJA::RangeSegment{0, n},
                               [=](int i, int j) {
                                 A->at(i, j) = (double)((i + j) % n) / (5 * m);
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

static void kernel_atax(int m,
                        int n,
                        const Arr2D<double>* A,
                        const Arr1D<double>* x,
                        Arr1D<double>* y,
                        Arr1D<double>* tmp) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, std::max(m, n), [=](int i) {
    if (i < n) y->at(i) = 0;
    if (i < m) tmp->at(i) = 0;
  });
  RAJA::forallN<OuterIndependent2D>(RAJA::RangeSegment{0, m},
                                    RAJA::RangeSegment{0, n},
                                    [=](int i, int j) {
                                      tmp->at(i) += A->at(i, j) * x->at(j);
                                    });
  RAJA::forallN<OuterIndependent2D>(RAJA::RangeSegment{0, n},
                                    RAJA::RangeSegment{0, m},
                                    [=](int j, int i) {
                                      y->at(j) += A->at(i, j) * tmp->at(i);
                                    });
#pragma endscop
}

int main(int argc, char** argv) {
  int m = M;
  int n = N;
  Arr2D<double> A{m, n};
  Arr1D<double> x{n};
  Arr1D<double> y{n};
  Arr1D<double> tmp{m};

  init_array(m, n, &A, &x);
  {
    util::block_timer t{"ATAX"};
    kernel_atax(m, n, &A, &x, &y, &tmp);
  }
  if (argc > 42) print_array(n, &y);
  return 0;
}
