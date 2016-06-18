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

#define ACC_1D(arr,s1,i1) (*(arr+(i1)))
#define ACC_2D(arr,s1,s2,i1,i2) (*(arr+(s2)*(i1)+(i2)))
#define ACC_3D(arr,s1,s2,s3,i1,i2,i3) (*(arr+(s3)*((s2)*(i1)+(i2))+(i3)))
#define ACC_4D(arr,s1,s2,s3,s4,i1,i2,i3,i4) (*(arr+(s4)*((s3)*((s2)*(i1)+(i2))+(i3))+(i4)))

template <typename T>
using Ptr = T* __restrict__;

template <typename T>
using CPtr = const Ptr<T>;


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


#include <array>

template <typename T, size_t N>
class MultiDimArray {
  std::array<size_t,N> extents;
  Ptr<T> data;

  Ptr<T> calculateSize () {
    size_t allocSize { 1 };
    for (int i { 0 }; i < N; ++i)
      allocSize *= extents[i];
    return static_cast <Ptr<T>> (polybench_alloc_data(allocSize, sizeof(T)));
  }

  size_t computeOffset (const std::array <size_t, N> &loc) const {
    size_t offset { loc[0] };
    for (int i = 1; i < N; ++i)
      offset = offset * extents[i] + loc[i];
    return offset;
  }

public:

  template <typename... DimensionLengths>
  MultiDimArray (DimensionLengths ... lengths) noexcept
    : extents {{static_cast<size_t>(lengths)...}},
      data { calculateSize () } { }

  ~MultiDimArray () {
    free (data);
  }

  template <typename... Ind>
  inline T& operator()(Ind&& ... indices) noexcept {
    return data [computeOffset({{static_cast<size_t>(std::forward<Ind>(indices))...}})];
  }

  template <typename... Ind>
  inline const T& operator()(Ind&& ... indices) const noexcept {
    return data [computeOffset({{static_cast<size_t>(std::forward<Ind>(indices))...}})];
  }
};

template<typename T> using Arr1D = MultiDimArray<T,1>;
template<typename T> using Arr2D = MultiDimArray<T,2>;
template<typename T> using Arr3D = MultiDimArray<T,3>;
template<typename T> using Arr4D = MultiDimArray<T,4>;

#endif /* !POLYBENCH_RAJA_HPP */
