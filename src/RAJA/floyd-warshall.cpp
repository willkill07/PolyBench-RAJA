/* floyd-warshall.c: this file is part of PolyBench/C */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "floyd-warshall.hpp"


static void init_array(int n, int path[N][N]) {
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i * j % 7 + 1;
      if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
        path[i][j] = 999;
    }
}

static void print_array(int n, int path[N][N]) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%d ", path[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "path");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_floyd_warshall(int n, int path[N][N]) {
  int i, j, k;
#pragma scop
  for (k = 0; k < n; k++) {
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++)
        path[i][j] = path[i][j] < path[i][k] + path[k][j]
                         ? path[i][j]
                         : path[i][k] + path[k][j];
  }
#pragma endscop
}

int main(int argc, char** argv) {
  int n = N;
  int(*path)[N][N];
  path = (int(*)[N][N])polybench_alloc_data((N) * (N), sizeof(int));
  init_array(n, *path);
  polybench_timer_start();
  kernel_floyd_warshall(n, *path);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], "")) print_array(n, *path);
  free((void*)path);
  return 0;
}
