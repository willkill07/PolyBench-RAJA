#ifndef _RAJA_JACOBI_2D_HPP_
#define _RAJA_JACOBI_2D_HPP_

#include <RAJA/RAJA.hxx>

#include "Base/jacobi-2d.hpp"

namespace RAJA {
template <typename T>
class jacobi_2d : public Base::jacobi_2d<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  jacobi_2d(Args... args) : Base::jacobi_2d<T>{"JACOBI-2D - RAJA", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A, B);
    using init_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;

    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        A->at(i, j) = (static_cast<T>(i) * (j + 2) + 2) / n;
        B->at(i, j) = (static_cast<T>(i) * (j + 3) + 3) / n;
      });
  }

  virtual void exec() {
    USE(READ, n, tsteps);
    USE(READWRITE, A, B);
    using exec_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;
    for (int t = 0; t < tsteps; ++t) {
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        [=](int i, int j) {
          B->at(i, j) =
            static_cast<T>(0.2)
            * (A->at(i, j) + A->at(i, j - 1) + A->at(i, 1 + j) + A->at(1 + i, j)
               + A->at(i - 1, j));
        });
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        [=](int i, int j) {
          A->at(i, j) =
            static_cast<T>(0.2)
            * (B->at(i, j) + B->at(i, j - 1) + B->at(i, 1 + j) + B->at(1 + i, j)
               + B->at(i - 1, j));
        });
    }
  }
};
} // RAJA
#endif
