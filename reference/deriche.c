/* deriche.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include <polybench.h>
/* Include benchmark-specific header. */
#include "deriche.h"

static void init_array(
  int w,
  int h,
  float *alpha,
  float imgIn[W][H],
  float imgOut[W][H])
{
  int i, j;
  *alpha = 0.25;
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++)
      imgIn[i][j] = (float)((313 * i + 991 * j) % 65536) / 65535.0f;
}

static void print_array(int w, int h, float imgOut[W][H])
{
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0)
        fprintf(stderr, "\n");
      fprintf(stderr, "%0.2f ", imgOut[i][j]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "imgOut");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_deriche(
  int w,
  int h,
  float alpha,
  float imgIn[W][H],
  float imgOut[W][H],
  float y1[W][H],
  float y2[W][H])
{
  int i, j;
  float xm1, tm1, ym1, ym2;
  float xp1, xp2;
  float tp1, tp2;
  float yp1, yp2;
  float k;
  float a1, a2, a3, a4, a5, a6, a7, a8;
  float b1, b2, c1, c2;
#pragma scop
  k = (1.0f - expf(-alpha)) * (1.0f - expf(-alpha))
      / (1.0f + 2.0f * alpha * expf(-alpha) - expf(2.0f * alpha));
  a1 = a5 = k;
  a2 = a6 = k * expf(-alpha) * (alpha - 1.0f);
  a3 = a7 = k * expf(-alpha) * (alpha + 1.0f);
  a4 = a8 = -k * expf(-2.0f * alpha);
  b1 = powf(2.0f, -alpha);
  b2 = -expf(-2.0f * alpha);
  c1 = c2 = 1;
  for (i = 0; i < w; i++) {
    ym1 = 0.0f;
    ym2 = 0.0f;
    xm1 = 0.0f;
    for (j = 0; j < h; j++) {
      y1[i][j] = a1 * imgIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
      xm1 = imgIn[i][j];
      ym2 = ym1;
      ym1 = y1[i][j];
    }
  }
  for (i = 0; i < w; i++) {
    yp1 = 0.0f;
    yp2 = 0.0f;
    xp1 = 0.0f;
    xp2 = 0.0f;
    for (j = h - 1; j >= 0; j--) {
      y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
      xp2 = xp1;
      xp1 = imgIn[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
    }
  for (j = 0; j < h; j++) {
    tm1 = 0.0f;
    ym1 = 0.0f;
    ym2 = 0.0f;
    for (i = 0; i < w; i++) {
      y1[i][j] = a5 * imgOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
      tm1 = imgOut[i][j];
      ym2 = ym1;
      ym1 = y1[i][j];
    }
  }
  for (j = 0; j < h; j++) {
    tp1 = 0.0f;
    tp2 = 0.0f;
    yp1 = 0.0f;
    yp2 = 0.0f;
    for (i = w - 1; i >= 0; i--) {
      y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
      tp2 = tp1;
      tp1 = imgOut[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++)
      imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
#pragma endscop
}

int main(int argc, char **argv)
{
  int w = W;
  int h = H;
  float alpha;
  float(*imgIn)[W][H];
  imgIn = (float(*)[W][H])polybench_alloc_data((W) * (H), sizeof(float));
  float(*imgOut)[W][H];
  imgOut = (float(*)[W][H])polybench_alloc_data((W) * (H), sizeof(float));
  float(*y1)[W][H];
  y1 = (float(*)[W][H])polybench_alloc_data((W) * (H), sizeof(float));
  float(*y2)[W][H];
  y2 = (float(*)[W][H])polybench_alloc_data((W) * (H), sizeof(float));
  init_array(w, h, &alpha, *imgIn, *imgOut);
  polybench_timer_start();
  kernel_deriche(w, h, alpha, *imgIn, *imgOut, *y1, *y2);
  polybench_timer_stop();
  polybench_timer_print();
  if (argc > 42 && !strcmp(argv[0], ""))
    print_array(w, h, *imgOut);
  free((void *)imgIn);
  free((void *)imgOut);
  free((void *)y1);
  free((void *)y2);
  return 0;
}
