/* gramschmidt.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "gramschmidt.hpp"

static void init_array(int m,
                       int n,
                       Arr2D<double>* A,
                       Arr2D<double>* R,
                       Arr2D<double>* Q) {
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, m},
                               RAJA::RangeSegment{0, n},
                               [=](int i, int j) {
                                 A->at(i, j) =
                                     (((double)((i * j) % m) / m) * 100) + 10;
                                 Q->at(i, j) = 0.0;
                               });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, n},
                               RAJA::RangeSegment{0, n},
                               [=](int i, int j) { R->at(i, j) = 0.0; });
}

static void print_array(int m,
                        int n,
                        const Arr2D<double>* R,
                        const Arr2D<double>* Q) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", R->at(i, j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "R");
  fprintf(stderr, "begin dump: %s", "Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", Q->at(i, j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "Q");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gramschmidt(int m,
                               int n,
                               Arr2D<double>* A,
                               Arr2D<double>* R,
                               Arr2D<double>* Q) {
#pragma scop
  RAJA::forall<RAJA::seq_exec>(0, n, [=](int k) {
    RAJA::ReduceSum<RAJA::omp_reduce, double> nrm{0.0};
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, m, [=](int i) {
      nrm += A->at(i, k) * A->at(i, k);
    });
    R->at(k, k) = sqrt(nrm);
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, m, [=](int i) {
      Q->at(i, k) = A->at(i, k) / R->at(k, k);
    });
    RAJA::forall<RAJA::omp_parallel_for_exec>(k + 1, n, [=](int j) {
      R->at(k, j) = 0.0;
      RAJA::forall<RAJA::simd_exec>(0, m, [=](int i) {
        R->at(k, j) += Q->at(i, k) * A->at(i, j);
      });
      RAJA::forall<RAJA::simd_exec>(0, m, [=](int i) {
        A->at(i, j) -= Q->at(i, k) * R->at(k, j);
      });
    });
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int m = M;
  int n = N;
  Arr2D<double> A{m, n}, R{n, n}, Q{m, n};

  init_array(m, n, &A, &R, &Q);
  {
    util::block_timer t{"GRAMSCHMIDT"};
    kernel_gramschmidt(m, n, &A, &R, &Q);
  }
  if (argc > 42) print_array(m, n, &R, &Q);
  return 0;
}
