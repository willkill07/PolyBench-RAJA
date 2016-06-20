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

extern void polybench_flush_cache();
extern void polybench_prepare_instruments();

#include <array>
#include <memory>

template <typename T>
struct DeleterWithFree {
  inline void operator() (T* ptr) {
    fprintf (stderr, "[MEM] Free-ing %p\n", (void*)ptr);
    free (ptr);
  }
};

template <typename T>
using Ptr = T* __restrict__;

template <typename T>
using CPtr = const Ptr<T>;

template <typename T>
using ManagedPtr = std::unique_ptr<T,DeleterWithFree<T>>;

template <typename T, size_t N>
class MultiDimArray {

  const std::array<size_t,N> extents;
  const std::array<size_t,N> coeffs;
  ManagedPtr<T> data;
  Ptr<T> rawData;

  inline ManagedPtr<T> allocData () const noexcept {
    size_t allocSize { 1 };
    for (int i { 0 }; i < N; ++i)
      allocSize *= extents[i];
    void* d;
    posix_memalign (&d, 1024, allocSize * sizeof(T));
    fprintf (stderr, "[MEM] Alloc-ing %p\n", (void*)d);
    return { static_cast <Ptr<T>> (d), { } };
  }

  inline std::array<size_t,N> calculateCoeffs () const noexcept {
    std::array<size_t, N> res;
    size_t off { 1 };
    for (int i { N - 1 }; i >= 0; --i) {
      res [i] = off;
      off *= extents [i];
    }
    return res;
  }

  template <size_t Dim, typename Length, typename... Lengths>
  inline size_t computeOffset (Length curr, Lengths ... rest) const noexcept {
    return computeOffset<Dim+1> (rest ...) + extents[Dim] * curr;
  }

  template <size_t Dim, typename Length>
  inline size_t computeOffset (Length curr) const noexcept {
    return curr;
  }

public:

  MultiDimArray() = delete;

  template <typename... DimensionLengths>
  MultiDimArray (DimensionLengths ... lengths) noexcept
    : extents {{static_cast<size_t>(lengths)...}},
      coeffs { calculateCoeffs () },
      data { allocData () },
      rawData { data.get() } { }

  MultiDimArray<T,N> (const MultiDimArray<T,N> & rhs) noexcept
  : extents { rhs.extents },
    coeffs { rhs.coeffs },
    data { },
    rawData { rhs.rawData } { }

  template <typename... Ind>
  inline T& operator()(Ind... indices) noexcept {
    return rawData [computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline T& at(Ind... indices) noexcept {
    return rawData [computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline const T& operator()(Ind... indices) const noexcept {
    return rawData [computeOffset<0>(indices...)];
  }

  template <typename... Ind>
  inline const T& at(Ind... indices) const noexcept {
    return rawData [computeOffset<0>(indices...)];
  }
};

template<typename T> using Arr1D = MultiDimArray<T,1>;
template<typename T> using Arr2D = MultiDimArray<T,2>;
template<typename T> using Arr3D = MultiDimArray<T,3>;
template<typename T> using Arr4D = MultiDimArray<T,4>;

#endif /* !POLYBENCH_RAJA_HPP */
