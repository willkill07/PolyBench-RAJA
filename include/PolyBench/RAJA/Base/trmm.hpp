#ifndef _RAJA_BASE_TRMM_HPP_
#define _RAJA_BASE_TRMM_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/trmm.hpp"

namespace RAJA {
namespace Base {
template <typename T>
class trmm : public ::Base::trmm<T> {
  using Parent = ::Base::trmm<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  trmm(Args... args) : ::Base::trmm<T>{"TRMM - RAJA Base", args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, A, B);
    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<init_pol>(
      RangeSegment{0, m},
      RangeSegment{0, m},
      [=](int i, int j) {
        A->at(i, j) = ((j < i) ? (static_cast<T>((i + j) % m) / m) : (i == j));
      });
    forallN<init_pol>(
      RangeSegment{0, m},
      RangeSegment{0, n},
      [=](int i, int j) {
        B->at(i, j) = static_cast<T>((n + (i - j)) % n) / n;
      });
  }

  virtual void exec() {
    USE(READ, m, n, alpha, A);
    USE(READWRITE, B);
    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<init_pol>(
      RangeSegment{0, m},
      RangeSegment{0, n},
      [=](int i, int j) {
        forall<simd_exec>(i + 1, m, [=](int k) {
          B->at(i, j) += A->at(k, i) * B->at(k, j);
        });
        B->at(i, j) *= alpha;
      });
  }
};
} // Base
} // RAJA
#endif
