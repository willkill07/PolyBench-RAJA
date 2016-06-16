/* bicg.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "bicg.hpp"


static void init_array(int m, int n, double A[N][M], double r[N], double p[M]) {
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, m, [=] (int i) {
    p[i] = (double)(i % m) / m;
  });
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    r[i] = (double)(i % n) / n;
    RAJA::forall<RAJA::simd_exec> (0, m, [=] (int j) {
      A[i][j] = (double)(i * (j + 1) % n) / n;
    });
  });
}

static void print_array(int m, int n, double s[M], double q[N]) {
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
                        double A[N][M],
                        double s[M],
                        double q[N],
                        double p[M],
                        double r[N]) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, m, [=] (int i) {
    s[i] = 0.0;
  });
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, n, [=] (int i) {
    q[i] = 0.0;
  });
  RAJA::forallN<Independent2DTile<32,16>> (
    RAJA::RangeSegment{0,n},
    RAJA::RangeSegment{0,m},
    [=] (int i, int j) {
      s[j] += r[i] * A[i][j];
      q[i] += A[i][j] * p[j];
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int m = M;
  double(*A)[N][M];
  A = (double(*)[N][M])polybench_alloc_data((N) * (M), sizeof(double));
  double(*s)[M];
  s = (double(*)[M])polybench_alloc_data(M, sizeof(double));
  double(*q)[N];
  q = (double(*)[N])polybench_alloc_data(N, sizeof(double));
  double(*p)[M];
  p = (double(*)[M])polybench_alloc_data(M, sizeof(double));
  double(*r)[N];
  r = (double(*)[N])polybench_alloc_data(N, sizeof(double));
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
