/* nussinov.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "nussinov.h"
/* RNA bases represented as chars, range is [0,3] */
typedef char base;
#define match(b1, b2) (((b1) + (b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

static void init_array(int n, base seq[N], int table[N][N]) {
  int i, j;
  for (i = 0; i < n; i++) {
    seq[i] = (base)((i + 1) % 4);
  }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      table[i][j] = 0;
}

static void print_array(int n, int table[N][N]) {
  int i, j;
  int t = 0;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "table");
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0)
        fprintf(stderr, "\n");
      fprintf(stderr, "%d ", table[i][j]);
      t++;
    }
  }
  fprintf(stderr, "\nend   dump: %s\n", "table");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}
# 62 "nussinov.c"

static void kernel_nussinov(int n, base seq[N], int table[N][N]) {
  int i, j, k;
#pragma scop
  for (i = n - 1; i >= 0; i--) {
    for (j = i + 1; j < n; j++) {
      if (j - 1 >= 0)
        table[i][j] =
          ((table[i][j] >= table[i][j - 1]) ? table[i][j] : table[i][j - 1]);
      if (i + 1 < n)
        table[i][j] =
          ((table[i][j] >= table[i + 1][j]) ? table[i][j] : table[i + 1][j]);
      if (j - 1 >= 0 && i + 1 < n) {
        if (i < j - 1)
          table[i][j] =
            ((table[i][j]
              >= table[i + 1][j - 1] + (((seq[i]) + (seq[j])) == 3 ? 1 : 0))
               ? table[i][j]
               : table[i + 1][j - 1] + (((seq[i]) + (seq[j])) == 3 ? 1 : 0));
        else
          table[i][j] =
            ((table[i][j] >= table[i + 1][j - 1]) ? table[i][j]
                                                  : table[i + 1][j - 1]);
      }
      for (k = i + 1; k < j; k++) {
        table[i][j] =
          ((table[i][j] >= table[i][k] + table[k + 1][j])
             ? table[i][j]
             : table[i][k] + table[k + 1][j]);
      }
    }
  }
#pragma endscop
}

int main(int argc, char **argv) {
  int n = N;
  base(*seq)[N];
  seq = (base(*)[N])polybench_alloc_data(N, sizeof(base));
  int(*table)[N][N];
  table = (int(*)[N][N])polybench_alloc_data((N) * (N), sizeof(int));
  init_array(n, *seq, *table);
  polybench_timer_start();
  kernel_nussinov(n, *seq, *table);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], ""))
    print_array(n, *table);
  free((void *)seq);
  free((void *)table);
  return 0;
}
