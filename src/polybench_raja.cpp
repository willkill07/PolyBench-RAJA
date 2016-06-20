#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "polybench_raja.hpp"

struct PolybenchData {
  double program_total_flops = 0;
  union {
    double time;
    unsigned long long int cycles;
  } start, end;
} _PolybenchData;

static double rtclock() {
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, NULL);
  if (stat != 0) printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

static unsigned long long int rdtsc() {
  unsigned long long int ret = 0;
  unsigned int cycles_lo;
  unsigned int cycles_hi;
  __asm__ volatile("RDTSC" : "=a"(cycles_lo), "=d"(cycles_hi));
  ret = (unsigned long long int)cycles_hi << 32 | cycles_lo;

  return ret;
}

void polybench_flush_cache() {
  int cs = POLYBENCH_CACHE_SIZE_KB * 1024 / sizeof(double);
  double* flush = (double*)calloc(cs, sizeof(double));
  int i;
  double tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : tmp) private(i)
#endif
  for (i = 0; i < cs; i++)
    tmp += flush[i];
  assert(tmp <= 10.0);
  free(flush);
}

void polybench_prepare_instruments() {
  polybench_flush_cache();
}

void polybench_timer_start() {
  polybench_prepare_instruments();
#ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
  _PolybenchData.start.time = rtclock();
#else
  _PolybenchData.start.cycles = rdtsc();
#endif
}

void polybench_timer_stop() {
#ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
  _PolybenchData.end.time = rtclock();
#else
  _PolybenchData.end.cycles = rdtsc();
#endif
}

void polybench_timer_print() {
#ifdef POLYBENCH_GFLOPS
  if (polybench_program_total_flops == 0) {
    printf(
        "[PolyBench][WARNING] Program flops not defined, use "
        "polybench_set_program_flops(value)\n");
    printf("%0.6lf\n", polybench_t_end - polybench_t_start);
  } else
    printf("%0.2lf\n",
           (polybench_program_total_flops
            / (double)(polybench_t_end - polybench_t_start))
               / 1000000000);
#else
#ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
  printf("%0.6f\n", _PolybenchData.end.time - _PolybenchData.start.time);
#else
  printf("%Ld\n", _PolybenchData.end.cycles - _PolybenchData.start.cycles);
#endif
#endif
}
