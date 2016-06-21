/* correlation.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "correlation.hpp"

static void init_array(int m, int n, double* float_n, Arr2D<double>* data) {
  *float_n = (double)N;
  RAJA::forallN<Independent2D> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      data->at(i, j) = (double)(i * j) / m + i;
    }
  );
}

static void print_array(int m, Arr2D<double>* corr) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", corr->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "corr");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_correlation(int m,
                               int n,
                               double float_n,
                               Arr2D<double>* data,
                               Arr2D<double>* corr,
                               Arr1D<double>* mean,
                               Arr1D<double>* stddev) {
  double eps = 0.1;
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, m, [=] (int j) {
    mean->at(j) = 0.0;
    RAJA::forall<RAJA::simd_exec> (0, n, [=] (int i) {
      mean->at(j) += data->at(i, j);
    });
    mean->at(j) /= float_n;
  });
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, m, [=] (int j) {
    stddev->at(j) = 0.0;
    RAJA::forall<RAJA::simd_exec> (0, n, [=] (int i) {
      stddev->at(j) += (data->at(i, j) - mean->at(j)) * (data->at(i, j) - mean->at(j));
    });
    stddev->at(j) = sqrt (stddev->at(j) / float_n);
    stddev->at(j) = (stddev->at(j) <= eps) ? 1.0 : stddev->at(j);
  });
  RAJA::forallN<Independent2DTiledVerbose<32,16>> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      data->at(i, j) = (data->at(i, j) - mean->at(j)) / sqrt(float_n) * stddev->at(j);
      corr->at(i, j) = (i == j) ? 1.0 : 0.0;
    }
  );
  RAJA::forallN<Independent2DTiledVerbose<32,16>> (
    RAJA::RangeSegment { 0, m - 1 },
    RAJA::RangeSegment { 1, m },
    [=] (int i, int j) {
      if (i < j) {
        corr->at(i, j) = 0.0;
        RAJA::forall<RAJA::simd_exec> (0, n, [=] (int k) {
            corr->at(i, j) += data->at(k, i) * data->at(k, j);
        });
        corr->at(j, i) = corr->at(i, j);
      }
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int m = M;
  double float_n;
  Arr2D<double> data { n, m };
  Arr2D<double> corr { m, m };
  Arr1D<double> mean { m };
  Arr1D<double> stddev { m };

  init_array(m, n, &float_n, &data);
  {
    util::block_timer t { "CORRELATION" };
    kernel_correlation(m, n, float_n, &data, &corr, &mean, &stddev);
  }
  if (argc > 42)
    print_array(m, &corr);
  return 0;
}
