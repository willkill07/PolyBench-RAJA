/* trmm.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "trmm.hpp"

static void init_array(int m,
                       int n,
                       double* alpha,
                       Arr2D<double>* A,
                       Arr2D<double>* B) {
  *alpha = 1.5;
  RAJA::forallN<Independent2D>(
      RAJA::RangeSegment{0, m}, RAJA::RangeSegment{0, m}, [=](int i, int j) {
        A->at(i, j) = ((j < i) ? ((double)((i + j) % m) / m) : (i == j));
      });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, m},
                               RAJA::RangeSegment{0, n},
                               [=](int i, int j) {
                                 B->at(i, j) = (double)((n + (i - j)) % n) / n;
                               });
}

static void print_array(int m, int n, const Arr2D<double>* B) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "B");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", B->at(i, j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "B");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_trmm(int m,
                        int n,
                        double alpha,
                        const Arr2D<double>* A,
                        Arr2D<double>* B) {
#pragma scop
  RAJA::forallN<Independent2D>(
      RAJA::RangeSegment{0, m}, RAJA::RangeSegment{0, n}, [=](int i, int j) {
        RAJA::forall<RAJA::simd_exec>(i + i, m, [=](int k) {
          B->at(i, j) += A->at(k, i) * B->at(k, j);
        });
        B->at(i, j) *= alpha;
      });
#pragma endscop
}

int main(int argc, char** argv) {
  int m = M;
  int n = N;
  double alpha;
  Arr2D<double> A{m, n}, B{m, n};

  init_array(m, n, &alpha, &A, &B);
  {
    util::block_timer t{"TRMM"};
    kernel_trmm(m, n, alpha, &A, &B);
  }
  if (argc > 42) print_array(m, n, &B);
  return 0;
}
