/* seidel-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "seidel-2d.h"

static void init_array(int n, double A[2000][2000]) {
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((double)i * (j + 2) + 2) / n;
}

static void print_array(int n, double A[2000][2000])

{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", A[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_seidel_2d(int tsteps, int n, double A[2000][2000]) {
  int t, i, j;

#pragma scop
  for (t = 0; t <= tsteps - 1; t++)
    for (i = 1; i <= n - 2; i++)
      for (j = 1; j <= n - 2; j++)
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1]
                   + A[i][j]
                   + A[i][j + 1]
                   + A[i + 1][j - 1]
                   + A[i + 1][j]
                   + A[i + 1][j + 1])
                  / 9.0;
#pragma endscop
}

int main(int argc, char** argv) {
  int n = 2000;
  int tsteps = 500;

  double(*A)[2000][2000];
  A = (double(*)[2000][2000])polybench_alloc_data((2000) * (2000),
                                                  sizeof(double));

  init_array(n, *A);

  polybench_timer_start();

  kernel_seidel_2d(tsteps, n, *A);

  polybench_timer_stop();
  polybench_timer_print();

  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *A);

  free((void*)A);

  return 0;
}
