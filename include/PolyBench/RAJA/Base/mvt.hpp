#ifndef _RAJA_BASE_MVT_HPP_
#define _RAJA_BASE_MVT_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/mvt.hpp"

namespace RAJA {
namespace Base {
template <typename T>
class mvt : public ::Base::mvt<T> {
  using Parent = ::Base::mvt<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  mvt(Args... args) : ::Base::mvt<T>{"MVT - RAJA Base", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A, x1, x2, y_1, y_2);
    forall<simd_exec>(0, n, [=](int i) {
      x1->at(i) = static_cast<T>(i % n) / n;
      x2->at(i) = static_cast<T>((i + 1) % n) / n;
      y_1->at(i) = static_cast<T>((i + 3) % n) / n;
      y_2->at(i) = static_cast<T>((i + 4) % n) / n;
      forall<simd_exec>(0, n, [=](int j) {
        A->at(i, j) = static_cast<T>(i * j % n) / n;
      });
    });
  }

  virtual void exec() {
    USE(READ, n, A, y_1, y_2);
    USE(READWRITE, x1, x2);
    using pol_exec = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<pol_exec>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) { x1->at(i) += A->at(i, j) * y_1->at(j); });
    forallN<pol_exec>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) { x2->at(i) += A->at(j, i) * y_2->at(j); });
  }
};
} // Base
} // RAJA
#endif
