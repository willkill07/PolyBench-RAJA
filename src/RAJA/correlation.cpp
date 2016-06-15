/* correlation.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "correlation.hpp"


static void init_array(int m, int n, double* float_n, double data[N][M]) {
  int i, j;
  *float_n = (double)N;
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (double)(i * j) / M + i;
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
  int i, j, k;
  double eps = 0.1;
#pragma scop
  for (j = 0; j < m; j++) {
    mean[j] = 0.0;
    for (i = 0; i < n; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }
  for (j = 0; j < m; j++) {
    stddev[j] = 0.0;
    for (i = 0; i < n; i++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    stddev[j] /= float_n;
    stddev[j] = sqrt(stddev[j]);
    stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
  }
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }
  for (i = 0; i < m - 1; i++) {
    corr[i][i] = 1.0;
    for (j = i + 1; j < m; j++) {
      corr[i][j] = 0.0;
      for (k = 0; k < n; k++)
        corr[i][j] += (data[k][i] * data[k][j]);
      corr[j][i] = corr[i][j];
    }
  }
  corr[m - 1][m - 1] = 1.0;
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
