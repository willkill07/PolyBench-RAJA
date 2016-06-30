#ifndef _RAJA_OMP_FDTD_2D_HPP_
#define _RAJA_OMP_FDTD_2D_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/fdtd-2d.hpp"

namespace RAJA
{
namespace OpenMP
{
template <typename T>
class fdtd_2d : public ::Base::fdtd_2d<T>
{
public:
  template <typename... Args>
  fdtd_2d(Args... args) : ::Base::fdtd_2d<T>{"FDTD-2D - RAJA OpenMP", args...}
  {
  }
  virtual void init()
  {
    USE(READ, _fict_, tmax, nx, ny);
    USE(READWRITE, ex, ey, hz);
    using init_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    forall<simd_exec>(0, tmax, [=](int i) {
      _fict_->at(i) = static_cast<T>(i);
    });
    forallN<init_pol>(
      RangeSegment{0, nx},
      RangeSegment{0, ny},
      [=](int i, int j) {
        ex->at(i, j) = (static_cast<T>(i) * (j + 1)) / nx;
        ey->at(i, j) = (static_cast<T>(i) * (j + 2)) / ny;
        hz->at(i, j) = (static_cast<T>(i) * (j + 3)) / nx;
      });
  }

  virtual void exec()
  {
    USE(READ, nx, ny, _fict_, tmax);
    USE(READWRITE, ex, ey, hz);
    using exec_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;

    for (int t = 0; t < tmax; ++t) {
      forall<simd_exec>(0, ny, [=](int j) { ey->at(0, j) = _fict_->at(t); });
      forallN<exec_pol>(
        RangeSegment{1, nx},
        RangeSegment{0, ny},
        [=](int i, int j) {
          ey->at(i, j) = ey->at(i, j) - 0.5 * (hz->at(i, j) - hz->at(i - 1, j));
        });
      forallN<exec_pol>(
        RangeSegment{0, nx},
        RangeSegment{1, ny},
        [=](int i, int j) {
          ex->at(i, j) = ex->at(i, j) - 0.5 * (hz->at(i, j) - hz->at(i, j - 1));
        });
      forallN<exec_pol>(
        RangeSegment{0, nx - 1},
        RangeSegment{0, ny - 1},
        [=](int i, int j) {
          hz->at(i, j) =
            hz->at(i, j)
            - 0.7 * (ex->at(i, j + 1) - ex->at(i, j) + ey->at(i + 1, j)
                     - ey->at(i, j));
        });
    }
  }
};
} // OpenMP
} // RAJA
#endif
