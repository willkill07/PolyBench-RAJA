/*
 * polybench_raja.hpp: this file is part of PolyBench/C
 *
 * Polybench header for instrumentation.
 *
 * Programs must be compiled with `-I utilities utilities/polybench.c'
 *
 * Optionally, one can define:
 *
 * -DPOLYBENCH_TIME, to report the execution time,
 *   OR (exclusive):
 * -DPOLYBENCH_PAPI, to use PAPI H/W counters (defined in polybench.c)
 *
 *
 * See README or utilities/polybench.c for additional options.
 *
 */
#ifndef POLYBENCH_RAJA_HPP
#define POLYBENCH_RAJA_HPP

#ifndef POLYBENCH_CACHE_SIZE_KB
#define POLYBENCH_CACHE_SIZE_KB 32770
#endif

#ifndef POLYBENCH_CACHE_LINE_SIZE_B
#define POLYBENCH_CACHE_LINE_SIZE_B 64
#endif

#include <RAJA/RAJA.hxx>

#include "MultiDimArray.hpp"

#include "PolyBenchKernel.hpp"

#endif /* !POLYBENCH_RAJA_HPP */
