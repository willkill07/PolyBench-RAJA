/* covariance.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"

static void init_array(int m, int n, double* float_n, double data[1400][1200]) {
  int i, j;

  *float_n = (double)n;

  for (i = 0; i < 1400; i++)
    for (j = 0; j < 1200; j++)
      data[i][j] = ((double)i * j) / 1200;
}

static void print_array(int m, double cov[1200][1200])

{
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
                              double data[1400][1200],
                              double cov[1200][1200],
                              double mean[1200]) {
  int i, j, k;

#pragma scop
  for (j = 0; j < m; j++) {
    mean[j] = 0.0;
    for (i = 0; i < n; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < m; i++)
    for (j = i; j < m; j++) {
      cov[i][j] = 0.0;
      for (k = 0; k < n; k++)
        cov[i][j] += data[k][i] * data[k][j];
      cov[i][j] /= (float_n - 1.0);
      cov[j][i] = cov[i][j];
    }
#pragma endscop
}

int main(int argc, char** argv) {
  int n = 1400;
  int m = 1200;

  double float_n;
  double(*data)[1400][1200];
  data = (double(*)[1400][1200])polybench_alloc_data((1400) * (1200),
                                                     sizeof(double));
  double(*cov)[1200][1200];
  cov = (double(*)[1200][1200])polybench_alloc_data((1200) * (1200),
                                                    sizeof(double));
  double(*mean)[1200];
  mean = (double(*)[1200])polybench_alloc_data(1200, sizeof(double));

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
