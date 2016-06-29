#ifndef _RAJA_BASE_LU_HPP_
#define _RAJA_BASE_LU_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/lu.hpp"

namespace RAJA {
namespace Base {
template <typename T>
class lu : public ::Base::lu<T> {
  using Parent = ::Base::lu<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  lu(Args... args) : ::Base::lu<T>{"LU - RAJA Base", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A);
    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    using init_pol_3 = NestedPolicy<ExecList<simd_exec, simd_exec, simd_exec>>;
    Arr2D<T> *B = new Arr2D<T>{n, n};
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        A->at(i, j) = (j < i) ? static_cast<T>(-j % n) / n + 1 : (i == j);
      });
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int s) { B->at(r, s) = 0; });
    forallN<init_pol_3>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int t, int s) { B->at(r, s) += A->at(r, t) * A->at(s, t); });
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int s) { A->at(r, s) = B->at(r, s); });
    delete B;
  }

  virtual void exec() {
    USE(READ, n);
    USE(READWRITE, A);
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        forall<simd_exec>(0, std::min(j, i), [=](int k) {
          A->at(i, j) -= A->at(i, k) * A->at(k, j);
        });
        if (j < i)
          A->at(i, j) /= A->at(j, j);
      });
  }
};
} // Base
} // RAJA
#endif
