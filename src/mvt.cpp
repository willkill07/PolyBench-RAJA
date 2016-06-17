/* mvt.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "mvt.hpp"

static void init_array(int n,
                       Arr1D<double>* x1,
                       Arr1D<double>* x2,
                       Arr1D<double>* y_1,
                       Arr1D<double>* y_2,
                       Arr2D<double>* A) {
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, n, [=](int i) {
    x1->at(i) = (double)(i % n) / n;
    x2->at(i) = (double)((i + 1) % n) / n;
    y_1->at(i) = (double)((i + 3) % n) / n;
    y_2->at(i) = (double)((i + 4) % n) / n;
    RAJA::forall<RAJA::simd_exec>(0, n, [=](int j) {
      A->at(i, j) = (double)(i * j % n) / n;
    });
  });
}

static void print_array(int n,
                        const Arr1D<double>* x1,
                        const Arr1D<double>* x2) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", x1->at(i));
  }
  fprintf(stderr, "\nend   dump: %s\n", "x1");
  fprintf(stderr, "begin dump: %s", "x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", x2->at(i));
  }
  fprintf(stderr, "\nend   dump: %s\n", "x2");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_mvt(int n,
                       Arr1D<double>* x1,
                       Arr1D<double>* x2,
                       const Arr1D<double>* y_1,
                       const Arr1D<double>* y_2,
                       const Arr2D<double>* A) {
#pragma scop
  RAJA::forallN<OuterIndependent2D>(RAJA::RangeSegment{0, n},
                                    RAJA::RangeSegment{0, n},
                                    [=](int i, int j) {
                                      x1->at(i) += A->at(i, j) * y_1->at(j);
                                    });
  RAJA::forallN<OuterIndependent2D>(RAJA::RangeSegment{0, n},
                                    RAJA::RangeSegment{0, n},
                                    [=](int i, int j) {
                                      x2->at(i) += A->at(j, i) * y_2->at(j);
                                    });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  Arr2D<double> A{n, n};
  Arr1D<double> x1{n}, x2{n}, y_1{n}, y_2{n};

  init_array(n, &x1, &x2, &y_1, &y_2, &A);
  {
    util::block_timer t{"MVT"};
    kernel_mvt(n, &x1, &x2, &y_1, &y_2, &A);
  }
  if (argc > 42) print_array(n, &x1, &x2);
  return 0;
}
