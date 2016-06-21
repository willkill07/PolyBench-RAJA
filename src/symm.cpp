/* symm.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include "polybench_raja.hpp"
/* Include benchmark-specific header. */
#include "symm.hpp"

static void init_array(int m,
                       int n,
                       double *alpha,
                       double *beta,
                       Arr2D<double>* C,
                       Arr2D<double>* A,
                       Arr2D<double>* B) {
  *alpha = 1.5;
  *beta = 1.2;
  RAJA::forallN<Independent2DTiled> (
    RAJA::RangeSegment { 0, m },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
      C->at(i,j) = (double)((i + j) % 100) / m;
      B->at(i,j) = (double)((n + i - j) % 100) / m;
      A->at(i,j) = (j <= i) ? ((double)((i + j) % 100) / m) : (-999);
    }
  );
}

static void print_array(int m, int n, const Arr2D<double>* C) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", C->at(i,j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_symm(int m,
                        int n,
                        double alpha,
                        double beta,
                        Arr2D<double>* C,
                        Arr2D<double>* A,
                        Arr2D<double>* B) {
#pragma scop
  RAJA::forallN<
    RAJA::NestedPolicy<
      RAJA::ExecList<RAJA::simd_exec,RAJA::omp_parallel_for_exec>,
      RAJA::Tile<RAJA::TileList<RAJA::tile_fixed<16>,RAJA::tile_fixed<16> >,
                 RAJA::Permute<RAJA::PERM_JI>>
    >
  > (
    RAJA::RangeSegment { 0, m },
    RAJA::RangeSegment { 0, n },
    [=] (int i, int j) {
			RAJA::ReduceSum <RAJA::seq_reduce, double> temp2 { 0.0 };
      RAJA::forall<RAJA::simd_exec> (0, i, [=] (int k) mutable {
        C->at(k,j) += alpha * B->at(i,j) * A->at(i,k);
        temp2 += B->at(k,j) * A->at(i,k);
      });
      C->at(i,j) = beta * C->at(i,j) + alpha * B->at(i,j) * A->at(i,i) + alpha * temp2;
    }
  );
#pragma endscop
}

int main(int argc, char &&argv) {
  int m = M;
  int n = N;
  double alpha;
  double beta;
	Arr2D<double> C { m, n }, A { m, m }, B { m, n };

	init_array(m, n, &alpha, &beta, &C, &A, &B);
	{
		util::block_timer t { "SYMM" };
		kernel_symm(m, n, alpha, beta, &C, &A, &B);
	}
	if (argc > 42)
		print_array(m, n, &C);
  return 0;
}
