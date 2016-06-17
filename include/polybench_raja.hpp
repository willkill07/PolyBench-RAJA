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

#include <cstdlib>

#define RAJA_ENABLE_NESTED 1
#include <RAJA/RAJA.hxx>

#include <hyper_array.hpp>

using OMP_ParallelRegion = typename RAJA::NestedPolicy<
  RAJA::ExecList<
    RAJA::seq_exec
  >,
  RAJA::OMP_Parallel<RAJA::Execute>
>;

using OuterIndependent2D = typename RAJA::NestedPolicy<
  RAJA::ExecList<
    RAJA::omp_parallel_for_exec,
    RAJA::simd_exec
  >,
  RAJA::Tile<
    RAJA::TileList<
      RAJA::tile_fixed<16>,
      RAJA::tile_none
    >,
    RAJA::Permute<RAJA::PERM_IJ>
  >
>;

using Independent2D = typename RAJA::NestedPolicy<
  RAJA::ExecList<
    RAJA::omp_collapse_nowait_exec,
    RAJA::omp_collapse_nowait_exec
  >,
  RAJA::OMP_Parallel<RAJA::Execute>
>;

using Independent3D = typename RAJA::NestedPolicy<
  RAJA::ExecList<
    RAJA::omp_collapse_nowait_exec,
    RAJA::omp_collapse_nowait_exec,
    RAJA::omp_collapse_nowait_exec
  >,
  RAJA::OMP_Parallel<RAJA::Execute>
>;

template <size_t Loop1 = 32, size_t Loop2 = 16, typename Permutation = RAJA::PERM_IJ>
using Independent2DTiledVerbose = typename RAJA::NestedPolicy<
  RAJA::ExecList<
    RAJA::omp_collapse_nowait_exec,
    RAJA::omp_collapse_nowait_exec
  >,
  RAJA::OMP_Parallel<
    RAJA::Tile<
      RAJA::TileList<
        RAJA::tile_fixed<Loop1>,
        RAJA::tile_fixed<Loop2>
      >,
      RAJA::Permute<Permutation>
    >
  >
>;

using Independent2DTiled = Independent2DTiledVerbose<>;

template <size_t Loop1 = 16, size_t Loop2 = 16, size_t Loop3 = 16, typename Permutation = RAJA::PERM_IJK>
using Independent3DTiledVerbose = typename RAJA::NestedPolicy<
  RAJA::ExecList<
    RAJA::omp_collapse_nowait_exec,
    RAJA::omp_collapse_nowait_exec,
    RAJA::omp_collapse_nowait_exec
  >,
  RAJA::OMP_Parallel<
    RAJA::Tile<
      RAJA::TileList<
        RAJA::tile_fixed<Loop1>,
        RAJA::tile_fixed<Loop2>,
        RAJA::tile_fixed<Loop3>
      >,
      RAJA::Permute<Permutation>
    >
  >
>;

using Independent3DTiled = Independent3DTiledVerbose<>;

extern double polybench_program_total_flops;
extern void polybench_timer_start();
extern void polybench_timer_stop();
extern void polybench_timer_print();

extern void* polybench_alloc_data(unsigned long long int n, int elt_size);
extern void polybench_free_data(void* ptr);

extern void polybench_flush_cache();
extern void polybench_prepare_instruments();

template <typename T> using Arr1D = hyper_array::array<T, 1>;
template <typename T> using Arr2D = hyper_array::array<T, 2>;
template <typename T> using Arr3D = hyper_array::array<T, 3>;
template <typename T> using Arr4D = hyper_array::array<T, 4>;
template <typename T> using Arr5D = hyper_array::array<T, 5>;

#endif /* !POLYBENCH_RAJA_HPP */
