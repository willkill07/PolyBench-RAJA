/* fdtd-2d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "fdtd-2d.hpp"

static void init_array(int tmax,
                       int nx,
                       int ny,
                       Arr2D<double>* ex,
                       Arr2D<double>* ey,
                       Arr2D<double>* hz,
                       Arr1D<double>* _fict_) {
  RAJA::forall<RAJA::omp_parallel_for_exec> (0, tmax, [=] (int i) {
    _fict_->at(i) = (double)i;
  });
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, nx },
    RAJA::RangeSegment { 0, ny },
    [=] (int i, int j) {
      ex->at(i,j) = ((double)i * (j + 1)) / nx;
      ey->at(i,j) = ((double)i * (j + 2)) / ny;
      hz->at(i,j) = ((double)i * (j + 3)) / nx;
    }
  );
}

static void print_array(int nx,
                        int ny,
                        const Arr2D<double>* ex,
                        const Arr2D<double>* ey,
                        const Arr2D<double>* hz) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", ex->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "ex");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", ey->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "ey");
  fprintf(stderr, "begin dump: %s", "hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", hz->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "hz");
}

static void kernel_fdtd_2d(int tmax,
                           int nx,
                           int ny,
                           Arr2D<double>* ex,
                           Arr2D<double>* ey,
                           Arr2D<double>* hz,
                           const Arr1D<double>* _fict_) {
#pragma scop
  RAJA::forall <RAJA::seq_exec> (0, tmax, [=] (int t) {
    RAJA::forall<RAJA::omp_parallel_for_exec> (0, ny, [=] (int j) {
      ey->at(0,j) = _fict_->at(t);
    });
    RAJA::forallN<Independent2DTiled> (
      RAJA::RangeSegment { 1, nx },
      RAJA::RangeSegment { 0, ny },
      [=] (int i, int j) {
        ey->at(i,j) = ey->at(i,j) - 0.5 * (hz->at(i,j) - hz->at(i - 1,j));
      }
    );
    RAJA::forallN<Independent2DTiled> (
      RAJA::RangeSegment { 1, nx },
      RAJA::RangeSegment { 1, ny },
      [=] (int i, int j) {
        ex->at(i,j) = ex->at(i,j) - 0.5 * (hz->at(i,j) - hz->at(i,j - 1));
      }
    );
    RAJA::forallN<Independent2DTiled> (
      RAJA::RangeSegment { 1, nx - 1 },
      RAJA::RangeSegment { 1, ny - 1 },
      [=] (int i, int j) {
        hz->at(i,j) = hz->at(i,j) - 0.7 * (ex->at(i,j + 1) - ex->at(i,j) + ey->at(i + 1,j) - ey->at(i,j));
      }
    );
	});
#pragma endscop
}

int main(int argc, char** argv) {
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;
	Arr2D<double> ex { nx, ny }, ey { nx, ny }, hz { nx, ny };
	Arr1D<double> _fict_ { tmax };
  init_array(tmax, nx, ny, &ex, &ey, &hz, &_fict_);

  {
		util::block_timer t { "FDTD-2D" };
		kernel_fdtd_2d(tmax, nx, ny, &ex, &ey, &hz, &_fict_);
  }
	if (argc > 42)
    print_array(nx, ny, &ex, &ey, &hz);
  return 0;
}
