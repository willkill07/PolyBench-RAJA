#ifndef _RAJA_MVT_HPP_
#define _RAJA_MVT_HPP_

#include <RAJA/RAJA.hxx>

#include "Base/mvt.hpp"

namespace RAJA {
template <typename T>
class mvt : public Base::mvt<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 1>::type>
  mvt(Args... args) : Base::mvt<T>{"MVT - RAJA", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A, x1, x2, y_1, y_2);
    forall<omp_parallel_for_exec>(0, n, [=](int i) {
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
    using pol_exec = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<32>, tile_none>>>;
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
} // RAJA
#endif
