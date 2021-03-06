#ifndef _RAJA_OMP_FLOYD_WARSHALL_HPP_
#define _RAJA_OMP_FLOYD_WARSHALL_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/floyd-warshall.hpp"

namespace RAJA
{
namespace OpenMP
{
template <typename T>
class floyd_warshall : public ::Base::floyd_warshall<T>
{
  using Parent = ::Base::floyd_warshall<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  floyd_warshall(Args... args)
  : ::Base::floyd_warshall<T>{"FLOYD-WARSHALL - RAJA OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, path);
    using init_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        path->at(i, j) = i * j % 7 + 1;
        if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
          path->at(i, j) = 999;
      });
  }

  virtual void exec()
  {
    USE(READ, n);
    USE(READWRITE, path);
    using exec_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>,
                                 tile_fixed<16>,
                                 tile_fixed<16>>>>;
    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int k, int i, int j) {
        path->at(i, j) =
          std::min(path->at(i, j), path->at(i, k) + path->at(k, j));
      });
  }
};
} // OpenMP
} // RAJA
#endif
