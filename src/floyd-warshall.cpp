/* floyd-warshall.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "floyd-warshall.hpp"

static void init_array(int n, Arr2D<int>* path) {
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, n },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      path->at(i,j) = i * j % 7 + 1;
      if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
        path->at(i,j) = 999;
    }
  );
}

static void print_array(int n, const Arr2D<int>* path) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%d ", path->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "path");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_floyd_warshall(int n, Arr2D<int>* path) {
#pragma scop
  RAJA::forallN<
    RAJA::NestedPolicy<
      RAJA::ExecList<
        RAJA::omp_parallel_for_exec,
        RAJA::seq_exec,
        RAJA::seq_exec>>>
    (RAJA::RangeSegment { 0, n },
     RAJA::RangeSegment { 0, n },
     RAJA::RangeSegment { 0, n },
     [=] (int k, int i, int j) {
       path->at(i,j) = std::min (path->at(i,j), path->at(i,k) + path->at(k,j));
     }
  );
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
	Arr2D<int> path { n, n };

  init_array(n, &path);
  {
		util::block_timer t { "FLOYD-WARSHALL" };
		kernel_floyd_warshall(n, &path);
	}
	if (argc > 42)
		print_array(n, &path);
  return 0;
}
