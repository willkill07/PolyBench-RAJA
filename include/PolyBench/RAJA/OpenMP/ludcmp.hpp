#ifndef _RAJA_OMP_LUDCMP_HPP_
#define _RAJA_OMP_LUDCMP_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/ludcmp.hpp"

namespace RAJA {
namespace OpenMP {
template <typename T>
class ludcmp : public ::Base::ludcmp<T> {
  using Parent = ::Base::ludcmp<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  ludcmp(Args... args) : ::Base::ludcmp<T>{"LUDCMP - RAJA OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, n, fn);
    USE(READWRITE, A, x, y, b);
    Arr2D<T> *B = new Arr2D<T>{n, n};
    forall<simd_exec>(0, n, [=](int i) {
      x->at(i) = static_cast<T>(0);
      y->at(i) = static_cast<T>(0);
      b->at(i) = (i + 1) / fn / static_cast<T>(2.0) + 4;
    });
    forallN<NestedPolicy<ExecList<simd_exec, simd_exec>>>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) {
        A->at(i, j) = (j < i) ? (static_cast<T>(-j % n) / n + 1) : (i == j);
      });
    forallN<NestedPolicy<ExecList<simd_exec, simd_exec>>>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int s) { B->at(r, s) = 0; });
    forallN<NestedPolicy<ExecList<simd_exec, simd_exec, simd_exec>>>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int s, int t) { B->at(r, s) += A->at(r, t) * A->at(s, t); });
    forallN<NestedPolicy<ExecList<simd_exec, simd_exec>>>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int r, int s) { A->at(r, s) = B->at(r, s); });
    delete B;
  }

  virtual void exec() {
    USE(READ, n);
    USE(READWRITE, A, y, x, b);
    forall<simd_exec>(0, n, [=](int i) {
      forall<simd_exec>(0, i, [=](int j) {
        ReduceSum<seq_reduce, T> w{-A->at(i, j)};
        forall<simd_exec>(0, j, [=](int k) { w += A->at(i, k) * A->at(k, j); });
        A->at(i, j) = -w / A->at(j, j);
      });
      forall<simd_exec>(i, n, [=](int j) {
        ReduceSum<seq_reduce, T> w{-A->at(i, j)};
        forall<simd_exec>(0, i, [=](int k) { w += A->at(i, k) * A->at(k, j); });
        A->at(i, j) = -w;
      });
    });
    forall<simd_exec>(0, n, [=](int i) {
      ReduceSum<seq_reduce, T> w{-b->at(i)};
      forall<simd_exec>(0, i, [=](int j) { w += A->at(i, j) * y->at(j); });
      y->at(i) = -w;
    });
    forall<simd_exec>(0, n, [=](int i_) {
      int i = n - (i_ + 1);
      ReduceSum<seq_reduce, T> w{-y->at(i)};
      forall<simd_exec>(i + 1, n, [=](int j) { w += A->at(i, j) * x->at(j); });
      x->at(i) = -w / A->at(i, i);
    });
  }
};
} // OpenMP
} // RAJA
#endif
