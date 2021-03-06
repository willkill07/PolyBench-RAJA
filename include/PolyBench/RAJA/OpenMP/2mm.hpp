#ifndef _RAJA_OMP_2MM_HPP_
#define _RAJA_OMP_2MM_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/2mm.hpp"

namespace RAJA
{
namespace OpenMP
{
template <typename T>
class mm2 : public ::Base::mm2<T>
{
  using Parent = ::Base::mm2<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  mm2(Args... args) : ::Base::mm2<T>{std::string{"2MM - RAJA OpenMP"}, args...}
  {
  }

  virtual void init()
  {
    USE(READWRITE, A, B, C, D);
    USE(READ, ni, nj, nk, nl);

    using init_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<32>, tile_fixed<32>>>>;
    forallN<init_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nk},
      [=](int i, int k) {
        A->at(i, k) = static_cast<T>((i * k + 1) % ni) / ni;
      });

    forallN<init_pol>(
      RangeSegment{0, nk},
      RangeSegment{0, ni},
      [=](int k, int i) {
        B->at(k, i) = static_cast<T>(k * (i + 1) % ni) / ni;
      });

    forallN<init_pol>(
      RangeSegment{0, nj},
      RangeSegment{0, nl},
      [=](int j, int l) {
        C->at(j, l) = static_cast<T>((j * (l + 3) + 1) % nl) / nl;
      });

    forallN<init_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nl},
      [=](int i, int l) {
        D->at(i, l) = static_cast<T>(i * (l + 2) % nk) / nk;
      });
  }

  virtual void exec()
  {
    USE(READWRITE, D, tmp);
    USE(READ, ni, nj, nk, nl, A, B, C, alpha, beta);

    using exec_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<32>, tile_fixed<32>>>>;
    forallN<exec_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nj},
      [=](int i, int j) {
        tmp->at(i, j) = static_cast<T>(0.0);
        forall<simd_exec>(0, nk, [=](int k) {
          tmp->at(i, j) += alpha * A->at(i, k) * B->at(k, j);
        });
      });
    forallN<exec_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nl},
      [=](int i, int l) {
        D->at(i, l) *= beta;
        forall<simd_exec>(0, nj, [=](int j) {
          D->at(i, l) += tmp->at(i, j) * C->at(j, l);
        });
      });
  }
};
} // OpenMP
} // RAJA
#endif
