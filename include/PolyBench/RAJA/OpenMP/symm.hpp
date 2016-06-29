#ifndef _RAJA_OMP_SYMM_HPP_
#define _RAJA_OMP_SYMM_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/symm.hpp"

namespace RAJA {
namespace OpenMP {
template <typename T>
class symm : public ::Base::symm<T> {
  using Parent = ::Base::symm<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  symm(Args... args) : ::Base::symm<T>{"SYMM - RAJA OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, A, C, B);
    using init_pol =
      NestedPolicy<ExecList<simd_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    forallN<init_pol>(
      RangeSegment{0, m},
      RangeSegment{0, n},
      [=](int i, int j) {
        C->at(i, j) = static_cast<T>((i + j) % 100) / m;
        B->at(i, j) = static_cast<T>((n + i - j) % 100) / m;
      });
    forallN<init_pol>(
      RangeSegment{0, m},
      RangeSegment{0, m},
      [=](int i, int j) {
        A->at(i, j) = (j > i) ? -999 : (static_cast<T>((i + j) % 100) / m);
      });
  }

  virtual void exec() {
    USE(READ, m, n, alpha, beta, A, B);
    USE(READWRITE, C);
    using exec_pol =
      NestedPolicy<ExecList<simd_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    forallN<exec_pol>(
      RangeSegment{0, m},
      RangeSegment{0, n},
      [=](int i, int j) {
        ReduceSum<seq_reduce, T> temp2{0.0};
        forall<simd_exec>(0, i, [=](int k) {
          C->at(k, j) += alpha * B->at(i, j) * A->at(i, k);
          temp2 += B->at(k, j) * A->at(i, k);
        });
        C->at(i, j) = beta * C->at(i, j) + alpha * B->at(i, j) * A->at(i, i)
                      + alpha * temp2;
      });
  }
};
} // OpenMP
} // RAJA
#endif
