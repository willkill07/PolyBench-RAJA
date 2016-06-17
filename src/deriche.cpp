/* deriche.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "deriche.hpp"

static void init_array(int w, int h, float* alpha, Arr2D<float>* imgIn) {
  *alpha = 0.25;
  RAJA::forallN<Independent2D>(
      RAJA::RangeSegment{0, w}, RAJA::RangeSegment{0, h}, [=](int i, int j) {
        imgIn->at(i, j) = (float)((313 * i + 991 * j) % 65536) / 65535.0f;
      });
}

static void print_array(int w, int h, const Arr2D<float>* imgOut) {
  int i, j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2f ", imgOut->at(i, j));
    }
  fprintf(stderr, "\nend   dump: %s\n", "imgOut");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_deriche(int w,
                           int h,
                           float alpha,
                           const Arr2D<float>* imgIn,
                           Arr2D<float>* imgOut,
                           Arr2D<float>* y1,
                           Arr2D<float>* y2) {
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
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, w, [=](int i) {
    float _ym1{0.0f}, _ym2{0.0f}, _xm1{0.0f};
    float *ym1{&_ym1}, *ym2{&_ym2}, *xm1{&_xm1};
    RAJA::forall<RAJA::simd_exec>(0, h, [=](int j) {
      y1->at(i, j) = a1 * imgIn->at(i, j) + a2 * *xm1 + b1 * *ym1 + b2 * *ym2;
      *xm1 = imgIn->at(i, j);
      *ym2 = *ym1;
      *ym1 = y1->at(i, j);
    });
  });
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, w, [=](int i) {
    float _yp1{0.0f}, _yp2{0.0f}, _xp1{0.0f}, _xp2{0.0f};
    float *yp1{&_yp1}, *yp2{&_yp2}, *xp1{&_xp1}, *xp2{&_xp2};
    RAJA::forall<RAJA::simd_exec>(h - 1, -1, -1, [=](int j) {
      y2->at(i, j) = a3 * *xp1 + a4 * *xp2 + b1 * *yp1 + b2 * *yp2;
      *xp2 = *xp1;
      *xp1 = imgIn->at(i, j);
      *yp2 = *yp1;
      *yp1 = y2->at(i, j);
    });
  });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, w},
                               RAJA::RangeSegment{0, h},
                               [=](int i, int j) {
                                 imgOut->at(i, j) =
                                     c1 * (y1->at(i, j) + y2->at(i, j));
                               });
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, h, [=](int j) {
    float _ym1{0.0f}, _ym2{0.0f}, _tm1{0.0f};
    float *ym1{&_ym1}, *ym2{&_ym2}, *tm1{&_tm1};
    RAJA::forall<RAJA::simd_exec>(0, w, [=](int i) {
      y1->at(i, j) = a5 * imgOut->at(i, j) + a6 * *tm1 + b1 * *ym1 + b2 * *ym2;
      *tm1 = imgOut->at(i, j);
      *ym2 = *ym1;
      *ym1 = y1->at(i, j);
    });
  });
  RAJA::forall<RAJA::omp_parallel_for_exec>(0, h, [=](int j) {
    float _yp1{0.0f}, _yp2{0.0f}, _tp1{0.0f}, _tp2{0.0f};
    float *yp1{&_yp1}, *yp2{&_yp2}, *tp1{&_tp1}, *tp2{&_tp2};
    RAJA::forall<RAJA::simd_exec>(w - 1, -1, -1, [=](int i) {
      y2->at(i, j) = a7 * *tp1 + a8 * *tp2 + b1 * *yp1 + b2 * *yp2;
      *tp2 = *tp1;
      *tp1 = imgOut->at(i, j);
      *yp2 = *yp1;
      *yp1 = y2->at(i, j);
    });
  });
  RAJA::forallN<Independent2D>(RAJA::RangeSegment{0, w},
                               RAJA::RangeSegment{0, h},
                               [=](int i, int j) {
                                 imgOut->at(i, j) =
                                     c2 * (y1->at(i, j) + y2->at(i, j));
                               });
#pragma endscop
}

int main(int argc, char** argv) {
  int w = W;
  int h = H;
  float alpha;
  Arr2D<float> imgIn{w, h}, imgOut{w, h}, y1{w, h}, y2{w, h};

  init_array(w, h, &alpha, &imgIn);
  {
    util::block_timer t{"DERICHE"};
    kernel_deriche(w, h, alpha, &imgIn, &imgOut, &y1, &y2);
  }
  if (argc > 42) print_array(w, h, &imgOut);
  return 0;
}
