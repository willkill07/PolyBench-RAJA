#ifndef _RAJA_BICG_HPP_
#define _RAJA_BICG_HPP_

#include <RAJA/RAJA.hxx>

#include "Base/bicg.hpp"

namespace RAJA {
template <typename T>
class bicg : public Base::bicg<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  bicg(Args... args) : Base::bicg<T>{std::string{"BICG - RAJA"}, args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, p, r, A);
    forall<omp_parallel_for_exec>(0, std::max(m, n), [=](int i) {
      if (i < m)
        p->at(i) = static_cast<T>(i % m) / m;
      if (i < n)
        r->at(i) = static_cast<T>(i % n) / n;
    });
    using init_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;

    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, m},
      [=](int i, int j) { A->at(i, j) = static_cast<T>(i * (j + 1) % n) / n; });
  }

  virtual void exec() {
    USE(READ, m, n, A, r, p);
    USE(READWRITE, s, q);
    forall<omp_parallel_for_exec>(0, std::max(m, n), [=](int i) {
      if (i < m)
        s->at(i) = static_cast<T>(0.0);
      if (i < n)
        q->at(i) = static_cast<T>(0.0);
    });
    using exec_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;

    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, m},
      [=](int i, int j) { q->at(i) += A->at(i, j) * p->at(j); });
    forallN<exec_pol>(
      RangeSegment{0, m},
      RangeSegment{0, n},
      [=](int j, int i) { s->at(j) += r->at(i) * A->at(i, j); });
  }
};
} // RAJA
#endif
