/* gemm.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "gemm.hpp"

static void init_array(int ni,
                       int nj,
                       int nk,
                       double* alpha,
                       double* beta,
                       Arr2D<double>* C,
                       Arr2D<double>* A,
                       Arr2D<double>* B) {
  *alpha = 1.5;
  *beta = 1.2;
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, ni},
                               RAJA::RangeSegment{0, nj},
                               [=](int i, int j) {
                                 C->at(i, j) = (double)((i * j + 1) % ni) / ni;
                               });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, ni},
                               RAJA::RangeSegment{0, nk},
                               [=](int i, int j) {
                                 A->at(i, j) = (double)(i * (j + 1) % nk) / nk;
                               });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, nk},
                               RAJA::RangeSegment{0, nj},
                               [=](int i, int j) {
                                 B->at(i, j) = (double)(i * (j + 2) % nj) / nj;
                               });
}

static void print_array(int ni, int nj, const Arr2D<double>* C) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", C->at(i, j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gemm(int ni,
                        int nj,
                        int nk,
                        double alpha,
                        double beta,
                        Arr2D<double>* C,
                        const Arr2D<double>* A,
                        const Arr2D<double>* B) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, ni, [=](int i) {
    RAJA::forall<RAJA::simd_exec>(0, nj, [=](int j) { C->at(i, j) *= beta; });
    RAJA::forallN<RAJA::NestedPolicy<RAJA::ExecList<RAJA::simd_exec,
                                                    RAJA::simd_exec>,
                                     RAJA::Permute<RAJA::PERM_JI> > >(
        RAJA::RangeSegment{0, nk},
        RAJA::RangeSegment{0, nj},
        [=](int k, int j) {
          C->at(i, j) += alpha * A->at(i, k) * B->at(k, j);
        });
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  double alpha;
  double beta;
  Arr2D<double> C{ni, nj}, A{ni, nk}, B{nk, nj};

  init_array(ni, nj, nk, &alpha, &beta, &C, &A, &B);
  {
    util::block_timer t{"GEMM"};
    kernel_gemm(ni, nj, nk, alpha, beta, &C, &A, &B);
  }
  if (argc > 42) print_array(ni, nj, &C);
  return 0;
}
