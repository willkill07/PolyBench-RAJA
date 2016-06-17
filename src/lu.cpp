/* lu.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "lu.hpp"

static void init_array(int n, Arr2D<double>* A) {
  Arr2D<double> _B{n, n}, *B{&_B};
  RAJA::forallN<Independent2D>(
      RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=](int i, int j) {
        A->at(i, j) = (j < i) ? ((double)(-j % n) / n + 1) : (j == i);
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

static void print_array(int n, const Arr2D<double>* A) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", A->at(i, j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_lu(int n, Arr2D<double>* A) {
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
      A->at(i, j) /= A->at(j, j);
    });
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  Arr2D<double> A{n, n};

  init_array(n, &A);
  {
    util::block_timer t{"LU"};
    kernel_lu(n, &A);
  }
  if (argc > 42) print_array(n, &A);
  return 0;
}
