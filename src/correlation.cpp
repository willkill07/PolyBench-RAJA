/* correlation.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "correlation.hpp"


static void init_array(int m, int n, double* float_n, double data[N][M]) {
  *float_n = (double)N;
  RAJA::forallN<Independent2D> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      data[i][j] = (double)(i * j) / m + i;
    }
  );
}

static void print_array(int m, double corr[M][M]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", corr[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "corr");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_correlation(int m,
                               int n,
                               double float_n,
                               double data[N][M],
                               double corr[M][M],
                               double mean[M],
                               double stddev[M]) {
  double eps = 0.1;
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, m, [=] (int j) {
    mean[j] = 0.0;
    RAJA::forall<RAJA::simd_exec> (0, n, [=] (int i) {
      mean[j] += data[i][j];
    });
    mean[j] /= float_n;
  });
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, m, [=] (int j) {
    stddev[j] = 0.0;
    RAJA::forall<RAJA::simd_exec> (0, n, [=] (int i) {
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    });
    stddev[j] = sqrt (stddev[j] / float_n);
    stddev[j] = (stddev[j] <= eps) ? 1.0 : stddev[j];
  });
  RAJA::forallN<Independent2DTiledVerbose<32,16>> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, m },
    [=] (int i, int j) {
      data[i][j] = (data[i][j] - mean[j]) / sqrt(float_n) * stddev[j];
      corr[i][j] = (i == j) ? 1.0 : 0.0;
    }
  );
  RAJA::forallN<Independent2DTiledVerbose<32,16>> (
    RAJA::RangeSegment { 0, m - 1 },
    RAJA::RangeSegment { 1, m },
    [=] (int i, int j) {
      if (i < j) {
        corr[i][j] = 0.0;
        RAJA::forall<RAJA::simd_exec> (0, n, [=] (int k) {
            corr[i][j] += data[k][i] * data[k][j];
        });
        corr[j][i] = corr[i][j];
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
  double(*corr)[M][M];
  corr = (double(*)[M][M])polybench_alloc_data((M) * (M), sizeof(double));
  double(*mean)[M];
  mean = (double(*)[M])polybench_alloc_data(M, sizeof(double));
  double(*stddev)[M];
  stddev = (double(*)[M])polybench_alloc_data(M, sizeof(double));
  init_array(m, n, &float_n, *data);
  polybench_timer_start();
  kernel_correlation(m, n, float_n, *data, *corr, *mean, *stddev);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(m, *corr);
  free((void*)data);
  free((void*)corr);
  free((void*)mean);
  free((void*)stddev);
  return 0;
}
