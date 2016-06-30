#ifndef _RAJA_OMP_SEIDEL_2D_HPP_
#define _RAJA_OMP_SEIDEL_2D_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/seidel-2d.hpp"

namespace RAJA
{
namespace OpenMP
{
template <typename T>
class seidel_2d : public ::Base::seidel_2d<T>
{
  using Parent = ::Base::seidel_2d<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  seidel_2d(Args... args)
      : ::Base::seidel_2d<T>{"SEIDEL-2D - RAJA OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A);
    using init_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        A->at(i, j) = (static_cast<T>(i) * (j + 2) + 2) / n;
      });
  }

  virtual void exec()
  {
    USE(READ, n, tsteps);
    USE(READWRITE, A);
    using exec_pol =
      NestedPolicy<ExecList<simd_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    for (int t = 0; t < tsteps; ++t) {
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        [=](int i, int j) {
          A->at(i, j) =
            (A->at(i - 1, j - 1) + A->at(i - 1, j) + A->at(i - 1, j + 1)
             + A->at(i, j - 1)
             + A->at(i, j)
             + A->at(i, j + 1)
             + A->at(i + 1, j - 1)
             + A->at(i + 1, j)
             + A->at(i + 1, j + 1))
            / 9.0;
        });
    }
  }
};
} // OpenMP
} // RAJA
#endif
