#ifndef _RAJA_TRMM_HPP_
#define _RAJA_TRMM_HPP_

#include <RAJA/RAJA.hxx>

#include "Base/trmm.hpp"

namespace RAJA {
template <typename T>
class trmm : public Base::trmm<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  trmm(Args... args) : Base::trmm<T>{"TRMM - RAJA", args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, A, B);
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>, Execute>;
    forallN<exec_pol>(
      RangeSegment{0, m},
      RangeSegment{0, m},
      [=](int i, int j) {
        A->at(i, j) = ((j < i) ? (static_cast<T>((i + j) % m) / m) : (i == j));
      });
    forallN<exec_pol>(
      RangeSegment{0, m},
      RangeSegment{0, n},
      [=](int i, int j) {
        B->at(i, j) = static_cast<T>((n + (i - j)) % n) / n;
      });
  }

  virtual void exec() {
    USE(READ, m, n, alpha, A);
    USE(READWRITE, B);
    forall<simd_exec>(0, m, [=](int i) {
      forall<omp_parallel_for_exec>(0, n, [=](int j) {
        forall<simd_exec>(i + 1, m, [=](int k) {
          B->at(i, j) += A->at(k, i) * B->at(k, j);
        });
        B->at(i, j) *= alpha;
      });
    });
  }
};
} // RAJA
#endif
