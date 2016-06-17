/* covariance.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "covariance.hpp"

static void init_array(int m, int n, double* float_n, Arr2D<double>* data) {
  *float_n = (double)n;
  RAJA::forallN<Independent2DVerbose<32, 16>>(RAJA::RangeSegment{0, n},
                                              RAJA::RangeSegment{0, m},
                                              [=](int i, int j) {
                                                data->at(i, j) =
                                                    ((double)i * j) / m;
                                              });
}

static void print_array(int m, const Arr2D<double>* cov) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", cov->at(i, j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "cov");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_covariance(int m,
                              int n,
                              double float_n,
                              Arr2D<double>* data,
                              Arr2D<double>* cov,
                              Arr1D<double>* mean) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, m, [=](int j) {
    mean->at(j) = 0.0;
    RAJA::forall<RAJA::simd_exec>(0, n, [=](int i) {
      mean->at(j) += data->at(i, j);
    });
    mean->at(j) /= float_n;
  });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, n},
                               RAJA::RangeSegment{0, m},
                               [=](int i, int j) {
                                 data->at(i, j) -= mean->at(j);
                               });
  RAJA::forallN<Independent2D>(
      RAJA::RangeSegment{0, m}, RAJA::RangeSegment{0, m}, [=](int i, int j) {
        if (i < j) {
          cov->at(i, j) = 0.0;
          RAJA::forall<RAJA::simd_exec>(0, n, [=](int k) {
            cov->at(i, j) += data->at(k, i) * data->at(k, j);
          });
          cov->at(i, j) = cov->at(j, i) = cov->at(i, j) / (float_n - 1.0);
        }
      });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int m = M;
  double float_n;
  Arr2D<double> data{n, m};
  Arr2D<double> cov{m, m};
  Arr1D<double> mean{m};

  init_array(m, n, &float_n, &data);
  {
    util::block_timer t{"COVARIANCE"};
    kernel_covariance(m, n, float_n, &data, &cov, &mean);
  }
  if (argc > 42) print_array(m, &cov);
  return 0;
}
