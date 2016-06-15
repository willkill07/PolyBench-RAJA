/* bicg.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"

static void init_array(int m,
                       int n,
                       double A[2100][1900],
                       double r[2100],
                       double p[1900]) {
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (double)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (double)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (double)(i * (j + 1) % n) / n;
  }
}

static void print_array(int m, int n, double s[1900], double q[2100])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", s[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "s");
  fprintf(stderr, "begin dump: %s", "q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", q[i]);
  }
  fprintf(stderr, "\nend   dump: %s\n", "q");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_bicg(int m,
                        int n,
                        double A[2100][1900],
                        double s[1900],
                        double q[2100],
                        double p[1900],
                        double r[2100]) {
  int i, j;

#pragma scop
  for (i = 0; i < m; i++)
    s[i] = 0;
  for (i = 0; i < n; i++) {
    q[i] = 0.0;
    for (j = 0; j < m; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
    }
  }
#pragma endscop
}

int main(int argc, char** argv) {
  int n = 2100;
  int m = 1900;

  double(*A)[2100][1900];
  A = (double(*)[2100][1900])polybench_alloc_data((2100) * (1900),
                                                  sizeof(double));
  double(*s)[1900];
  s = (double(*)[1900])polybench_alloc_data(1900, sizeof(double));
  double(*q)[2100];
  q = (double(*)[2100])polybench_alloc_data(2100, sizeof(double));
  double(*p)[1900];
  p = (double(*)[1900])polybench_alloc_data(1900, sizeof(double));
  double(*r)[2100];
  r = (double(*)[2100])polybench_alloc_data(2100, sizeof(double));

  init_array(m, n, *A, *r, *p);

  polybench_timer_start();

  kernel_bicg(m, n, *A, *s, *q, *p, *r);

  polybench_timer_stop();
  polybench_timer_print();

  if (argc > 42 && !strcmp(argv[0], "")) print_array(m, n, *s, *q);

  free((void*)A);
  free((void*)s);
  free((void*)q);
  free((void*)p);
  free((void*)r);

  return 0;
}
