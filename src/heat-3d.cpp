/* heat-3d.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "heat-3d.hpp"

static void init_array(int n, Arr3D<double>* A, Arr3D<double>* B) {
  RAJA::forallN<Independent3D>(RAJA::RangeSegment{0, n},
                               RAJA::RangeSegment{0, n},
                               RAJA::RangeSegment{0, n},
                               [=](int i, int j, int k) {
                                 A->at(i, j, k) = B->at(i, j, k) =
                                     (double)(i + j + (n - k)) * 10 / (n);
                               });
}

static void print_array(int n, const Arr3D<double>* A) {
  int i, j, k;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
        if ((i * n * n + j * n + k) % 20 == 0) fprintf(stderr, "\n");
        fprintf(stderr, "%0.2lf ", A->at(i, j, k));
      }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_heat_3d(int tsteps,
                           int n,
                           Arr3D<double>* A,
                           Arr3D<double>* B) {
#pragma scop
  RAJA::forall<RAJA::seq_exec>(0, tsteps, [=](int t) {
    RAJA::forallN<Independent3D>(
        RAJA::RangeSegment{1, n - 1},
        RAJA::RangeSegment{1, n - 1},
        RAJA::RangeSegment{1, n - 1},
        [=](int i, int j, int k) {
          B->at(i, j, k) = 0.125 * (A->at(i + 1, j, k) - 2.0 * A->at(i, j, k)
                                    + A->at(i - 1, j, k))
                           + 0.125 * (A->at(i, j + 1, k) - 2.0 * A->at(i, j, k)
                                      + A->at(i, j - 1, k))
                           + 0.125 * (A->at(i, j, k + 1) - 2.0 * A->at(i, j, k)
                                      + A->at(i, j, k - 1))
                           + A->at(i, j, k);
        });
    RAJA::forallN<Independent3D>(
        RAJA::RangeSegment{1, n - 1},
        RAJA::RangeSegment{1, n - 1},
        RAJA::RangeSegment{1, n - 1},
        [=](int i, int j, int k) {
          A->at(i, j, k) = 0.125 * (B->at(i + 1, j, k) - 2.0 * B->at(i, j, k)
                                    + B->at(i - 1, j, k))
                           + 0.125 * (B->at(i, j + 1, k) - 2.0 * B->at(i, j, k)
                                      + B->at(i, j - 1, k))
                           + 0.125 * (B->at(i, j, k + 1) - 2.0 * B->at(i, j, k)
                                      + B->at(i, j, k - 1))
                           + B->at(i, j, k);
        });
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int tsteps = TSTEPS;
  Arr3D<double> A{n, n, n}, B{n, n, n};

  init_array(n, &A, &B);
  {
    util::block_timer t{"HEAT-3D"};
    kernel_heat_3d(tsteps, n, &A, &B);
  }
  if (argc > 42) print_array(n, &A);
  return 0;
}
