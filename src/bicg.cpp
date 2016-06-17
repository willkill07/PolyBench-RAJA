/* bicg.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "bicg.hpp"

static void init_array(int m,
                       int n,
                       Arr2D<double>* A,
                       Arr1D<double>* r,
                       Arr1D<double>* p) {
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, std::max(m, n), [=](int i) {
    if (i < m) p->at(i) = (double)(i % m) / m;
    if (i < n) r->at(i) = (double)(i % n) / n;
  });
  RAJA::forallN<OuterIndependent2D>(RAJA::RangeSegment{0, n},
                                    RAJA::RangeSegment{0, m},
                                    [=](int i, int j) {
                                      A->at(i, j) =
                                          (double)(i * (j + 1) % n) / n;
                                    });
}

static void print_array(int m,
                        int n,
                        const Arr1D<double>* s,
                        const Arr1D<double>* q) {
  int i;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", s->at(i));
  }
  fprintf(stderr, "\nend   dump: %s\n", "s");
  fprintf(stderr, "begin dump: %s", "q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf(stderr, "\n");
    fprintf(stderr, "%0.2lf ", q->at(i));
  }
  fprintf(stderr, "\nend   dump: %s\n", "q");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_bicg(int m,
                        int n,
                        const Arr2D<double>* A,
                        Arr1D<double>* s,
                        Arr1D<double>* q,
                        const Arr1D<double>* p,
                        const Arr1D<double>* r) {
#pragma scop
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, std::max(m, n), [=](int i) {
    if (i < m) s->at(i) = 0.0;
    if (i < n) q->at(i) = 0.0;
  });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, n},
                               RAJA::RangeSegment{0, m},
                               [=](int i, int j) {
                                 s->at(j) += r->at(i) * A->at(i, j);
                                 q->at(i) += A->at(i, j) * p->at(j);
                               });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int m = M;
  Arr2D<double> A{n, m};
  Arr1D<double> s{m};
  Arr1D<double> q{n};
  Arr1D<double> p{m};
  Arr1D<double> r{n};

  init_array(m, n, &A, &r, &p);
  {
    util::block_timer t{"BICG"};
    kernel_bicg(m, n, &A, &s, &q, &p, &r);
  }
  if (argc > 42) print_array(m, n, &s, &q);
  return 0;
}
