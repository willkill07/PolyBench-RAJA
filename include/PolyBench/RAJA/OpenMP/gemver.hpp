#ifndef _RAJA_OMP_GEMVER_HPP_
#define _RAJA_OMP_GEMVER_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/gemver.hpp"

namespace RAJA {
namespace OpenMP {
template <typename T>
class gemver : public ::Base::gemver<T> {
  using Parent = ::Base::gemver<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gemver(Args... args) : ::Base::gemver<T>{"GEMVER - RAJA OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, fn, n);
    USE(READWRITE, u1, u2, v1, v2, w, x, y, z, A);
    forall<omp_parallel_for_exec>(0, n, [=](int i) {
      u1->at(i) = i;
      u2->at(i) = ((i + 1) / fn) / 2.0;
      v1->at(i) = ((i + 1) / fn) / 4.0;
      v2->at(i) = ((i + 1) / fn) / 6.0;
      y->at(i) = ((i + 1) / fn) / 8.0;
      z->at(i) = ((i + 1) / fn) / 9.0;
      x->at(i) = static_cast<T>(0.0);
      w->at(i) = static_cast<T>(0.0);
      forall<simd_exec>(0, n, [=](int j) {
        A->at(i, j) = static_cast<T>(i * j % n) / n;
      });
    });
  }

  virtual void exec() {
    USE(READ, n, alpha, beta, u1, u2, v1, v2, y, z, A);
    USE(READWRITE, x, w);
    using exec_pol =
      NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                   Tile<TileList<tile_fixed<16>, tile_fixed<16>>>>;
    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        A->at(i, j) =
          A->at(i, j) + u1->at(i) * v1->at(j) + u2->at(i) * v2->at(j);
      });
    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        x->at(i) = x->at(i) + beta * A->at(j, i) * y->at(j);
      });
    forall<omp_parallel_for_exec>(0, n, [=](int i) {
      x->at(i) = x->at(i) + z->at(i);
    });
    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        w->at(i) = w->at(i) + alpha * A->at(i, j) * x->at(j);
      });
  }
};
} // OpenMP
} // RAJA
#endif
