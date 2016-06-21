/* seidel-2d.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "seidel-2d.hpp"

static void init_array(int n, Arr2D<double>* A) {
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      A->at(i,j) = ((double)i * (j + 2) + 2) / n;
    }
  );
}

static void print_array(int n, const Arr2D<double>* A) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", A->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_seidel_2d(int tsteps, int n, Arr2D<double>* A) {
#pragma scop
  RAJA::forallN<RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec,RAJA::simd_exec,RAJA::simd_exec>,
    RAJA::Tile<RAJA::TileList<RAJA::tile_none,RAJA::tile_fixed<16>,RAJA::tile_fixed<16>>>
  >> (
    RAJA::RangeSegment { 0, tsteps },
    RAJA::RangeSegment { 1, n - 1 },
    RAJA::RangeSegment { 1, n - 1 },
    [=] (int t, int i, int j) {
        A->at(i,j) = (A->at(i - 1,j - 1) + A->at(i - 1,j) + A->at(i - 1,j + 1) + A->at(i,j - 1)
                   + A->at(i,j)
                   + A->at(i,j + 1)
                   + A->at(i + 1,j - 1)
                   + A->at(i + 1,j)
                   + A->at(i + 1,j + 1))
                  / 9.0;
    }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int tsteps = 500;
	Arr2D<double> A { n, n };

  init_array(n, &A);
  {
		util::block_timer t { "SEIDEL-2D" };
		kernel_seidel_2d(tsteps, n, &A);
	}
  if (argc > 42)
		print_array(n, &A);
  return 0;
}
