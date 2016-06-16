/* covariance.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "covariance.hpp"


static void init_array(int m, int n, double* float_n, double data[N][M]) {
  *float_n = (double)n;
  RAJA::forallN<Independent2DTiledVerbose<32,16>> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      data[i][j] = ((double)i * j) / m;
    }
  );
}

static void print_array(int m, double cov[M][M]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", cov[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "cov");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_covariance(int m,
                              int n,
                              double float_n,
                              double data[N][M],
                              double cov[M][M],
                              double mean[M]) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, m, [=] (int j) {
    RAJA::ReduceSum<RAJA::omp_reduce, double> mean_r { 0.0 };
    RAJA::forall<RAJA::simd_exec> (0, n, [=] (int i) {
      mean_r += data[i][j];
    });
    mean[j] = mean_r / float_n;
  });
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      data[i][j] -= mean[j];
    }
  );
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, m },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      if (i < j) {
        RAJA::ReduceSum<RAJA::omp_reduce, double> cov_r { 0.0 };
        RAJA::forall<RAJA::simd_exec> (0, n, [=] (int k) {
          cov_r += data[k][i] * data[k][j];
        });
        cov[i][j] = cov[j][i] = cov_r / (float_n - 1.0);
      }
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int m = M;
  double float_n;
  double(*data)[N][M];
  data = (double(*)[N][M])polybench_alloc_data((N) * (M), sizeof(double));
  double(*cov)[M][M];
  cov = (double(*)[M][M])polybench_alloc_data((M) * (M), sizeof(double));
  double(*mean)[M];
  mean = (double(*)[M])polybench_alloc_data(M, sizeof(double));
  init_array(m, n, &float_n, *data);
  polybench_timer_start();
  kernel_covariance(m, n, float_n, *data, *cov, *mean);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(m, *cov);
  free((void*)data);
  free((void*)cov);
  free((void*)mean);
  return 0;
}
