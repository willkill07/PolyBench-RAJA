#ifndef _RAJA_OMP_ADI_HPP_
#define _RAJA_OMP_ADI_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/adi.hpp"

namespace RAJA {
namespace OpenMP {
template <typename T>
class adi : public ::Base::adi<T> {
  using Parent = ::Base::adi<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  adi(Args... args)
      : ::Base::adi<T>{std::string{"ADI - RAJA OpenMP"}, args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, u);

    using init_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;

    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) { u->at(i, j) = static_cast<T>(i + n - j) / n; });
  }

  virtual void exec() {
    USE(READ, n, tsteps);
    T DX(static_cast<T>(1.0) / n);
    T DY(static_cast<T>(1.0) / n);
    T DT(static_cast<T>(1.0) / tsteps);
    T B1(static_cast<T>(2.0));
    T B2(static_cast<T>(1.0));
    T mul1(B1 * DT / (DX * DX));
    T mul2(B2 * DT / (DY * DY));
    T a(-mul1 / static_cast<T>(2.0));
    T b(static_cast<T>(1.0) + mul1);
    T c(a);
    T d(-mul2 / static_cast<T>(2.0));
    T e(static_cast<T>(1.0) + mul2);
    T f(d);
    USE(READWRITE, v, u, p, q);
    using exec_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;
    for (int t = 0; t < tsteps; ++t) {
      forall<omp_parallel_for_exec>(1, n - 1, [=](int i) {
        v->at(0, i) = static_cast<T>(1.0);
        p->at(i, 0) = static_cast<T>(0.0);
        q->at(i, 0) = v->at(0, i);
        v->at(n - 1, i) = static_cast<T>(1.0);
      });
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        [=](int i, int j) {
          p->at(i, j) = -c / (a * p->at(i, j - 1) + b);
          q->at(i, j) = (-d * u->at(j, i - 1) + (1.0 + 2.0 * d) * u->at(j, i)
                         - f * u->at(j, i + 1)
                         - a * q->at(i, j - 1))
                        / (a * p->at(i, j - 1) + b);
        });
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{2, n},
        [=](int i, int j_) {
          int j = n - j_;
          v->at(j, i) = p->at(i, j) * v->at(j + 1, i) + q->at(i, j);
        });
      forall<omp_parallel_for_exec>(1, n - 1, [=](int i) {
        u->at(i, 0) = static_cast<T>(1.0);
        p->at(i, 0) = static_cast<T>(0.0);
        q->at(i, 0) = u->at(i, 0);
        u->at(i, n - 1) = static_cast<T>(1.0);
      });
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{1, n - 1},
        [=](int i, int j) {
          p->at(i, j) = -f / (d * p->at(i, j - 1) + e);
          q->at(i, j) =
            (-a * v->at(i - 1, j)
             + (static_cast<T>(1.0) + static_cast<T>(2.0) * a) * v->at(i, j)
             - c * v->at(i + 1, j)
             - d * q->at(i, j - 1))
            / (d * p->at(i, j - 1) + e);
        });
      forallN<exec_pol>(
        RangeSegment{1, n - 1},
        RangeSegment{2, n},
        [=](int i, int j_) {
          int j = n - j_;
          u->at(i, j) = p->at(i, j) * u->at(i, j + 1) + q->at(i, j);
        });
    }
  }
};
} // OpenMP
} // RAJA
#endif
