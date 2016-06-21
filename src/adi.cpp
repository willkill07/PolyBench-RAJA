/* adi.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "adi.hpp"


static void init_array(int n, Arr2D<double>* u) {
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      u->at(i,j) = (double)(i + n - j) / n;
    }
  );
}

static void print_array(int n, Arr2D<double>* u) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", u->at(i, j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "u");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_adi(int tsteps,
                       int n,
                       Arr2D<double>* u,
                       Arr2D<double>* v,
                       Arr2D<double>* p,
                       Arr2D<double>* q) {
  double DX, DY, DT;
  double B1, B2;
  double mul1, mul2;
  double a, b, c, d, e, f;
#pragma scop
  DX = 1.0 / (double)n;
  DY = 1.0 / (double)n;
  DT = 1.0 / (double)tsteps;
  B1 = 2.0;
  B2 = 1.0;
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);
  a = -mul1 / 2.0;
  b = 1.0 + mul1;
  c = a;
  d = -mul2 / 2.0;
  e = 1.0 + mul2;
  f = d;
  RAJA::forall<RAJA::seq_exec> (0, tsteps, [=] (int t) {
    RAJA::forall<RAJA::omp_parallel_for_exec> (1, n - 1, [=] (int i) {
      v->at(0, i) = 1.0;
      p->at(i, 0) = 0.0;
      q->at(i, 0) = v->at(0, i);
      v->at(n - 1, i) = 1.0;
    });
    RAJA::forallN<OuterIndependent2D> (
      RAJA::RangeSegment { 1, n - 1 },
      RAJA::RangeSegment { 1, n - 1 },
      [=] (int i, int j) {
        p->at(i, j) = -c / (a * p->at(i, j - 1) + b);
        q->at(i, j) =
          (-d *  u->at(j, i - 1) + (1.0 + 2.0 * d) * u->at(j, i) - f * u->at(j, i + 1)
             - a * q->at(i, j - 1))
            / (a * p->at(i, j - 1) + b);
      }
    );
    RAJA::forallN<OuterIndependent2D> (
      RAJA::RangeSegment { 1, n - 1 },
      RAJA::RangeStrideSegment { n - 2, 0, -1 },
      [=] (int i, int j) {
        v->at(j, i) = p->at(i, j) * v->at(j + 1, i) + q->at(i, j);
      }
    );
    RAJA::forall<RAJA::omp_parallel_for_exec> (1, n - 1, [=] (int i) {
      u->at(i, 0) = 1.0;
      p->at(i, 0) = 0.0;
      q->at(i, 0) = u->at(i, 0);
      u->at(i, n - 1) = 1.0;
    });
    RAJA::forallN<OuterIndependent2D> (
      RAJA::RangeSegment { 1, n - 1 },
      RAJA::RangeSegment { 1, n - 1 },
      [=] (int i, int j) {
        p->at(i, j) = -f / (d * p->at(i, j - 1) + e);
        q->at(i, j) =
          (-a * v->at(i - 1, j) + (1.0 + 2.0 * a) * v->at(i, j) - c * v->at(i + 1, j)
           - d * q->at(i, j - 1))
          / (d * p->at(i, j - 1) + e);
      }
    );
    RAJA::forallN<OuterIndependent2D> (
      RAJA::RangeSegment { 1, n - 1 },
      RAJA::RangeStrideSegment { n - 2, 0, -1 },
      [=] (int i, int j) {
        u->at(i, j) = p->at(i, j) * u->at(i, j + 1) + q->at(i, j);
      }
    );
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int tsteps = TSTEPS;
  Arr2D<double> u { n, n };
  Arr2D<double> v { n, n };
  Arr2D<double> p { n, n };
  Arr2D<double> q { n, n };

  init_array(n, &u);
  {
    util::block_timer t { "ADI" };
    kernel_adi(tsteps, n, &u, &v, &p, &q);
  }
  if (argc > 42)
    print_array(n, &u);
  return 0;
}
