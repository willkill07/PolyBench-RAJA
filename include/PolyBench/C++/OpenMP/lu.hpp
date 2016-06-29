#ifndef _CPP_OMP_LU_HPP_
#define _CPP_OMP_LU_HPP_

#include "PolyBench/Base/lu.hpp"

namespace CPlusPlus {
namespace OpenMP {
template <typename T>
class lu : public ::Base::lu<T> {
  using Parent = ::Base::lu<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  lu(Args... args) : ::Base::lu<T>{"LU - C++ OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A);
    Arr2D<T> *B = new Arr2D<T>{n, n};
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++)
        A->at(i, j) = static_cast<T>(-j % n) / n + 1;
      for (int j = i + 1; j < n; j++) {
        A->at(i, j) = 0;
      }
      A->at(i, i) = 1;
    }
    for (int r = 0; r < n; ++r)
      for (int s = 0; s < n; ++s)
        B->at(r, s) = 0;
    for (int t = 0; t < n; ++t)
      for (int r = 0; r < n; ++r)
        for (int s = 0; s < n; ++s)
          B->at(r, s) += A->at(r, t) * A->at(s, t);
    for (int r = 0; r < n; ++r)
      for (int s = 0; s < n; ++s)
        A->at(r, s) = B->at(r, s);
    delete B;
  }

  virtual void exec() {
    USE(READ, n);
    USE(READWRITE, A);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        for (int k = 0; k < j; k++) {
          A->at(i, j) -= A->at(i, k) * A->at(k, j);
        }
        A->at(i, j) /= A->at(j, j);
      }
      for (int j = i; j < n; j++) {
        for (int k = 0; k < i; k++) {
          A->at(i, j) -= A->at(i, k) * A->at(k, j);
        }
      }
    }
  }
};
} // OpenMP
} // CPlusPlus
#endif
