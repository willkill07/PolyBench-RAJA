/* fdtd-2d.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "fdtd-2d.h"

static void init_array(
  int tmax,
  int nx,
  int ny,
  double ex[NX][NY],
  double ey[NX][NY],
  double hz[NX][NY],
  double _fict_[TMAX]) {
  int i, j;
  for (i = 0; i < tmax; i++)
    _fict_[i] = (double)i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      ex[i][j] = ((double)i * (j + 1)) / nx;
      ey[i][j] = ((double)i * (j + 2)) / ny;
      hz[i][j] = ((double)i * (j + 3)) / nx;
    }
}

static void print_array(
  int nx,
  int ny,
  double ex[NX][NY],
  double ey[NX][NY],
  double hz[NX][NY]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0)
        fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", ex[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "ex");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0)
        fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", ey[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "ey");
  fprintf(stderr, "begin dump: %s", "hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0)
        fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", hz[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "hz");
}

static void kernel_fdtd_2d(
  int tmax,
  int nx,
  int ny,
  double ex[NX][NY],
  double ey[NX][NY],
  double hz[NX][NY],
  double _fict_[TMAX]) {
  int t, i, j;
#pragma scop
  for (t = 0; t < tmax; t++) {
    for (j = 0; j < ny; j++)
      ey[0][j] = _fict_[t];
    for (i = 1; i < nx; i++)
      for (j = 0; j < ny; j++)
        ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
    for (i = 0; i < nx; i++)
      for (j = 1; j < ny; j++)
        ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
    for (i = 0; i < nx - 1; i++)
      for (j = 0; j < ny - 1; j++)
        hz[i][j] =
          hz[i][j] - 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
  }
#pragma endscop
}

int main(int argc, char **argv) {
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;
  double(*ex)[NX][NY];
  ex = (double(*)[NX][NY])polybench_alloc_data((NX) * (NY), sizeof(double));
  double(*ey)[NX][NY];
  ey = (double(*)[NX][NY])polybench_alloc_data((NX) * (NY), sizeof(double));
  double(*hz)[NX][NY];
  hz = (double(*)[NX][NY])polybench_alloc_data((NX) * (NY), sizeof(double));
  double(*_fict_)[TMAX];
  _fict_ = (double(*)[TMAX])polybench_alloc_data(TMAX, sizeof(double));
  init_array(tmax, nx, ny, *ex, *ey, *hz, *_fict_);
  polybench_timer_start();
  kernel_fdtd_2d(tmax, nx, ny, *ex, *ey, *hz, *_fict_);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], ""))
    print_array(nx, ny, *ex, *ey, *hz) free((void *)ex);
  free((void *)ey);
  free((void *)hz);
  free((void *)_fict_);
  return 0;
}
