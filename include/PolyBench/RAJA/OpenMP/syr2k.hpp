#ifndef _RAJA_OMP_SYR2K_HPP_
#define _RAJA_OMP_SYR2K_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/syr2k.hpp"

namespace RAJA
{
namespace OpenMP
{
template <typename T>
class syr2k : public ::Base::syr2k<T>
{
  using Parent = ::Base::syr2k<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  syr2k(Args... args) : ::Base::syr2k<T>{"SYR2K - RAJA OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, m, n);
    USE(READWRITE, A, C, B);
    using init_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;

    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, m},
      [=](int i, int j) {
        A->at(i, j) = static_cast<T>((i * j + 1) % n) / n;
        B->at(i, j) = static_cast<T>((i * j + 2) % m) / m;
      });
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) { C->at(i, j) = static_cast<T>((i * j + 3) % n) / m; });
  }

  virtual void exec()
  {
    USE(READ, m, n, alpha, beta, A, B);
    USE(READWRITE, C);
    using exec_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        if (j <= i) {
          C->at(i, j) *= beta;
          forall<simd_exec>(0, m, [=](int k) {
            C->at(i, j) += A->at(j, k) * alpha * B->at(i, k)
                           + B->at(j, k) * alpha * A->at(i, k);
          });
        }
      });
  }
};
} // OpenMP
} // RAJA
#endif
