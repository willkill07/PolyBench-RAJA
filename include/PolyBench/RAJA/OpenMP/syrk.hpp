#ifndef _RAJA_OMP_SYRK_HPP_
#define _RAJA_OMP_SYRK_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/syrk.hpp"

namespace RAJA {
namespace OpenMP {
template <typename T>
class syrk : public ::Base::syrk<T> {
  using Parent = ::Base::syrk<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  syrk(Args... args) : ::Base::syrk<T>{"SYRK - RAJA OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, A, C);
    using init_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;

    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, m},
      [=](int i, int j) { A->at(i, j) = static_cast<T>((i * j + 1) % n) / n; });
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) { C->at(i, j) = static_cast<T>((i * j + 2) % m) / m; });
  }

  virtual void exec() {
    USE(READ, m, n, alpha, beta, A);
    USE(READWRITE, C);
    using exec_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;

    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        if (j <= i) {
          C->at(i, j) *= beta;
          forall<simd_exec>(0, m, [=](int k) {
            C->at(i, j) += alpha * A->at(i, k) * A->at(j, k);
          });
        }
      });
  }
};
} // OpenMP
} // RAJA
#endif
