#ifndef _RAJA_BASE_GEMM_HPP_
#define _RAJA_BASE_GEMM_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/gemm.hpp"

namespace RAJA {
namespace Base {
template <typename T>
class gemm : public ::Base::gemm<T> {
  using Parent = ::Base::gemm<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gemm(Args... args) : ::Base::gemm<T>{"GEMM - RAJA Base", args...} {
  }

  virtual void init() {
    USE(READ, ni, nj, nk);
    USE(READWRITE, A, B, C);
    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
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
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forall<simd_exec>(0, ni, [=](int i) {
      forall<simd_exec>(0, nj, [=](int j) { C->at(i, j) *= beta; });
      forallN<exec_pol>(
        RangeSegment{0, nk},
        RangeSegment{0, nj},
        [=](int k, int j) {
          C->at(i, j) += alpha * A->at(i, k) * B->at(k, j);
        });
    });
  }
};
} // Base
} // RAJA
#endif
