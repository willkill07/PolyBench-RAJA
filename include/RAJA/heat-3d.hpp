#ifndef _RAJA_HEAT_3D_HPP_
#define _RAJA_HEAT_3D_HPP_

#include <RAJA/RAJA.hxx>

#include "Base/heat-3d.hpp"

namespace RAJA {
template <typename T>
class heat_3d : public Base::heat_3d<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  heat_3d(Args... args) : Base::heat_3d<T>{"HEAT-3D - RAJA", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A, B);
    using init_pol = NestedPolicy<ExecList<omp_collapse_nowait_exec,
                                           omp_collapse_nowait_exec,
                                           simd_exec>,
                                  OMP_Parallel<Tile<TileList<tile_fixed<16>,
                                                             tile_fixed<16>,
                                                             tile_none>,
                                                    Execute>>>;
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j, int k) {
        A->at(i, j, k) = B->at(i, j, k) =
          static_cast<T>(i + j + (n - k)) * 10 / (n);
      });
  }

  virtual void exec() {
    USE(READ, n, tsteps);
    USE(READWRITE, A, B);
    using exec_pol = NestedPolicy<ExecList<omp_collapse_nowait_exec,
                                           omp_collapse_nowait_exec,
                                           simd_exec>,
                                  OMP_Parallel<Tile<TileList<tile_fixed<16>,
                                                             tile_fixed<16>,
                                                             tile_none>,
                                                    Execute>>>;
    for (int t = 0; t < tsteps; ++t) {
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        [=](int i, int j, int k) {
          B->at(i, j, k) =
            static_cast<T>(0.125)
              * (A->at(i + 1, j, k) - 2.0 * A->at(i, j, k) + A->at(i - 1, j, k))
            + static_cast<T>(0.125) * (A->at(i, j + 1, k) - 2.0 * A->at(i, j, k)
                                       + A->at(i, j - 1, k))
            + static_cast<T>(0.125) * (A->at(i, j, k + 1) - 2.0 * A->at(i, j, k)
                                       + A->at(i, j, k - 1))
            + A->at(i, j, k);
        });
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        [=](int i, int j, int k) {
          A->at(i, j, k) =
            static_cast<T>(0.125)
              * (B->at(i + 1, j, k) - 2.0 * B->at(i, j, k) + B->at(i - 1, j, k))
            + static_cast<T>(0.125) * (B->at(i, j + 1, k) - 2.0 * B->at(i, j, k)
                                       + B->at(i, j - 1, k))
            + static_cast<T>(0.125) * (B->at(i, j, k + 1) - 2.0 * B->at(i, j, k)
                                       + B->at(i, j, k - 1))
            + B->at(i, j, k);
        });
    }
  }
};
} // RAJA
#endif
