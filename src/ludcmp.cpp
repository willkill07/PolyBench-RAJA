/* ludcmp.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "ludcmp.hpp"

static void init_array(int n,
                       Arr2D<double>* A,
                       Arr1D<double>* b,
                       Arr1D<double>* x,
                       Arr1D<double>* y) {
  double fn = (double)n;
  Arr2D<double> _B{n, n}, *B{&_B};
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, n, [=](int i) {
    x->at(i) = 0;
    y->at(i) = 0;
    b->at(i) = (i + 1) / fn / 2.0 + 4;
  });
  RAJA::forallN<Independent2D>(
      RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=](int i, int j) {
        A->at(i, j) = (j < i) ? ((double)(-j % n) / n + 1) : (i == j);
      });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, n},
                               RAJA::RangeSegment{0, n},
                               [=](int r, int s) { B->at(r, s) = 0; });
  RAJA::forallN<Independent2D>(
      RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=](int t, int r) {
        RAJA::forall<RAJA::simd_exec>(0, n, [=](int s) {
          B->at(r, s) += A->at(r, t) * A->at(s, t);
        });
      });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, n},
                               RAJA::RangeSegment{0, n},
                               [=](int r, int s) {
                                 A->at(r, s) = B->at(r, s);
                               });
}

static void print_array(int n, const Arr1D<double>* x) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", x->at(i));
  }
  fprintf(stderr, "\nend   dump: %s\n", "x");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_ludcmp(int n,
                          Arr2D<double>* A,
                          Arr1D<double>* b,
                          Arr1D<double>* x,
                          Arr1D<double>* y) {
#pragma scop
  RAJA::forall<RAJA::seq_exec>(0, n, [=](int i) {
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, i, [=](int j) {
      RAJA::forall<RAJA::simd_exec>(0, j, [=](int k) {
        A->at(i, j) -= A->at(i, k) * A->at(k, j);
      });
      A->at(i, j) /= A->at(j, j);
    });
    RAJA::forall<RAJA::omp_parallel_for_exec>(i, n, [=](int j) {
      RAJA::forall<RAJA::simd_exec>(0, i, [=](int k) {
        A->at(i, j) -= A->at(i, k) * A->at(k, j);
      });
    });
  });
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, n, [=](int i) {
    RAJA::forall<RAJA::simd_exec>(0, i, [=](int j) {
      b->at(i) -= A->at(i, j) * y->at(j);
    });
    y->at(i) = b->at(i);
  });
  RAJA::forall<RAJA::omp_parallel_for_exec>(n - 1, -1, -1, [=](int i) {
    RAJA::forall<RAJA::simd_exec>(0, i, [=](int j) {
      y->at(i) -= A->at(i, j) * x->at(j);
    });
    x->at(i) = y->at(i) / A->at(i, i);
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  Arr2D<double> A{n, n};
  Arr1D<double> b{n}, x{n}, y{n};

  init_array(n, &A, &b, &x, &y);
  {
    util::block_timer t{"LUDCMP"};
    kernel_ludcmp(n, &A, &b, &x, &y);
  }
  if (argc > 42) print_array(n, &x);
  return 0;
}
