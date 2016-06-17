/* nussinov.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "nussinov.hpp"

/* RNA bases represented as chars, range is [0,3] */
typedef char base;
#define match(b1, b2) (((b1) + (b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

static void init_array(int n, Arr1D<base>* seq, Arr2D<int>* table) {
  RAJA::forall<RAJA::simd_exec>(0, n, [=](int i) {
    seq->at(i) = (base)((i + 1) % 4);
  });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, n},
                               RAJA::RangeSegment{0, n},
                               [=](int i, int j) { table->at(i, j) = 0; });
}

static void print_array(int n, const Arr2D<int>* table) {
  int i, j;
  int t = 0;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "table");
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%d ", table->at(i, j));
      t++;
    }
  }
  fprintf(stderr, "\nend   dump: %s\n", "table");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_nussinov(int n, const Arr1D<base>* seq, Arr2D<int>* table) {
#pragma scop
  RAJA::forall<RAJA::seq_exec>(n - 1, -1, -1, [=](int i) {
    RAJA::forall<RAJA::simd_exec>(i + 1, n, [=](int j) {
      if (j - 1 >= 0)
        table->at(i, j) = std::max(table->at(i, j), table->at(i, j - 1));
      if (i + 1 < n)
        table->at(i, j) = std::max(table->at(i, j), table->at(i + i, j));
      if (i < n - 1) {
        if (i < j - 1)
          table->at(i, j) = std::max(table->at(i, j),
                                     table->at(i + 1, j - 1)
                                         + ((seq->at(i) + seq->at(j)) == 3));
        else
          table->at(i, j) = std::max(table->at(i, j), table->at(i + 1, j - 1));
      }
      RAJA::forall<RAJA::simd_exec>(i + 1, j, [=](int k) {
        table->at(i, j) =
            std::max(table->at(i, j), table->at(i, k) + table->at(k + 1, j));
      });
    });
  });
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  Arr1D<base> seq{n};
  Arr2D<int> table{n, n};

  init_array(n, &seq, &table);
  {
    util::block_timer t{"NUSSINOV"};
    kernel_nussinov(n, &seq, &table);
  }
  if (argc > 42) print_array(n, &table);
  return 0;
}
