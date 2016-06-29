#ifndef _RAJA_OMP_GRAMSCHMIDT_HPP_
#define _RAJA_OMP_GRAMSCHMIDT_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/gramschmidt.hpp"

namespace RAJA {
namespace OpenMP {
template <typename T>
class gramschmidt : public ::Base::gramschmidt<T> {
  using Parent = ::Base::gramschmidt<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gramschmidt(Args... args)
      : ::Base::gramschmidt<T>{"GRAMSCHMIDT - RAJA OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, A, Q, R);
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<exec_pol>(
      RangeSegment{0, m},
      RangeSegment{0, n},
      [=](int i, int j) {
        A->at(i, j) = ((static_cast<T>((i * j) % m) / m) * 100) + 10;
        Q->at(i, j) = static_cast<T>(0.0);
      });
    forallN<exec_pol>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) { R->at(i, j) = static_cast<T>(0.0); });
  }

  virtual void exec() {
    USE(READ, m, n);
    USE(READWRITE, A, R, Q);
    forall<seq_exec>(0, n, [=](int k) {
      R->at(k, k) = static_cast<T>(0.0);
      forall<simd_exec>(0, m, [=](int i) {
        R->at(k, k) += A->at(i, k) * A->at(i, k);
      });
      R->at(k, k) = sqrt(R->at(k, k));
      forall<simd_exec>(0, m, [=](int i) {
        Q->at(i, k) = A->at(i, k) / R->at(k, k);
      });
      forall<simd_exec>(k + 1, n, [=](int j) {
        R->at(k, j) = static_cast<T>(0.0);
        forall<simd_exec>(0, m, [=](int i) {
          R->at(k, j) += Q->at(i, k) * A->at(i, j);
        });
        forall<simd_exec>(0, m, [=](int i) {
          A->at(i, j) = A->at(i, j) - Q->at(i, k) * R->at(k, j);
        });
      });
    });
  }
};
} // OpenMP
} // RAJA
#endif
