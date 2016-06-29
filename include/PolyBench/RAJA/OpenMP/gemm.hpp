#ifndef _RAJA_OMP_GEMM_HPP_
#define _RAJA_OMP_GEMM_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/gemm.hpp"

namespace RAJA {
namespace OpenMP {
template <typename T>
class gemm : public ::Base::gemm<T> {
  using Parent = ::Base::gemm<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gemm(Args... args) : ::Base::gemm<T>{"GEMM - RAJA OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, ni, nj, nk);
    USE(READWRITE, A, B, C);
    using init_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    forallN<init_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nj},
      [=](int i, int j) {
        C->at(i, j) = static_cast<T>((i * j + 1) % ni) / ni;
      });
    forallN<init_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nk},
      [=](int i, int j) {
        A->at(i, j) = static_cast<T>(i * (j + 1) % nk) / nk;
      });
    forallN<init_pol>(
      RangeSegment{0, nk},
      RangeSegment{0, nj},
      [=](int i, int j) {
        B->at(i, j) = static_cast<T>(i * (j + 2) % nj) / nj;
      });
  }

  virtual void exec() {
    USE(READ, ni, nj, nk, alpha, beta, A, B);
    USE(READWRITE, C);
    forall<omp_parallel_for_exec>(0, ni, [=](int i) {
      forall<simd_exec>(0, nj, [=](int j) { C->at(i, j) *= beta; });
      forallN<NestedPolicy<ExecList<simd_exec, simd_exec>>>(
        RangeSegment{0, nk},
        RangeSegment{0, nj},
        [=](int k, int j) {
          C->at(i, j) += alpha * A->at(i, k) * B->at(k, j);
        });
    });
  }
};
} // OpenMP
} // RAJA
#endif
