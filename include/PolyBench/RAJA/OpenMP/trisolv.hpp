#ifndef _RAJA_OMP_TRISOLV_HPP_
#define _RAJA_OMP_TRISOLV_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/trisolv.hpp"

namespace RAJA {
namespace OpenMP {
template <typename T>
class trisolv : public ::Base::trisolv<T> {
  using Parent = ::Base::trisolv<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  trisolv(Args... args) : ::Base::trisolv<T>{"TRISOLV - RAJA OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, x, b, L);
    forall<simd_exec>(0, n, [=](int i) {
      x->at(i) = -999;
      b->at(i) = i;
      forall<simd_exec>(0, i + 1, [=](int j) {
        L->at(i, j) = static_cast<T>(i + n - j + 1) * 2 / n;
      });
    });
  }

  virtual void exec() {
    USE(READ, n, b, L);
    USE(READWRITE, x);
    forall<simd_exec>(0, n, [=](int i) {
      ReduceSum<seq_reduce, T> v{0.0};
      forall<simd_exec>(0, i, [=](int j) { v += L->at(i, j) * x->at(j); });
      x->at(i) = (b->at(i) - v) / L->at(i, i);
    });
  }
};
} // OpenMP
} // RAJA
#endif
