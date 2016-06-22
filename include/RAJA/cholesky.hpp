#ifndef _RAJA_CHOLESKY_HPP_
#define _RAJA_CHOLESKY_HPP_

#include <RAJA/RAJA.hxx>

#include "Base/cholesky.hpp"

namespace RAJA {
template <typename T>
class cholesky : public Base::cholesky<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 1>::type>
  cholesky(Args... args)
      : Base::cholesky<T>{std::string{"CHOLESKY - RAJA"}, args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A);
    Arr2D<T> *B = new Arr2D<T>{n, n};
    forall<omp_parallel_for_exec>(0, n, [=](int i) {
      forall<simd_exec>(0, i + 1, [=](int j) {
        A->at(i, j) = static_cast<T>(-j % n) / n + static_cast<T>(1.0);
      });
      forall<simd_exec>(i + 1, n, [=](int j) {
        A->at(i, j) = static_cast<T>(0.0);
      });
      A->at(i, i) = 1;
    });
    using init_pol = NestedPolicy<ExecList<omp_parallel_for_exec, simd_exec>,
                                  Tile<TileList<tile_fixed<16>, tile_none>>>;
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int s) { B->at(r, s) = static_cast<T>(0); });
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int s) {
        forall<simd_exec>(0, n, [=](int t) {
          B->at(r, s) += A->at(r, t) * A->at(s, t);
        });
      });
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int s) { A->at(r, s) = B->at(r, s); });
    delete B;
  }

  virtual void exec() {
    USE(READ, n);
    USE(READWRITE, A);
    forall<simd_exec>(0, n, [=](int i) {
      forall<simd_exec>(0, i, [=](int j) {
        forall<simd_exec>(0, j, [=](int k) {
          A->at(i, j) -= A->at(i, k) * A->at(j, k);
        });
        A->at(i, j) /= A->at(j, j);
      });
      forall<simd_exec>(0, i, [=](int k) {
        A->at(i, i) -= A->at(i, k) * A->at(i, k);
      });
    });
    forall<omp_parallel_for_exec>(0, n, [=](int i) {
      A->at(i, i) = sqrt(A->at(i, i));
    });
  }
};
} // RAJA
#endif
