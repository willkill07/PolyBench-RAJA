#ifndef _CPP_OMP_LUDCMP_HPP_
#define _CPP_OMP_LUDCMP_HPP_

#include "PolyBench/Base/ludcmp.hpp"

namespace CPlusPlus {
namespace OpenMP {
template <typename T>
class ludcmp : public ::Base::ludcmp<T> {
  using Parent = ::Base::ludcmp<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  ludcmp(Args... args) : ::Base::ludcmp<T>{"LUDCMP - C++ OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, n, fn);
    USE(READWRITE, A, x, y, b);
    Arr2D<T> *B = new Arr2D<T>{n, n};
    for (int i = 0; i < n; i++) {
      x->at(i) = static_cast<T>(0);
      y->at(i) = static_cast<T>(0);
      b->at(i) = (i + 1) / fn / static_cast<T>(2.0) + 4;
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++)
        A->at(i, j) = static_cast<T>(-j % n) / n + 1;
      A->at(i, i) = 1;
      for (int j = i + 1; j < n; j++) {
        A->at(i, j) = 0;
      }
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
    USE(READWRITE, A, x, b, y);
    T w;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        w = A->at(i, j);
        for (int k = 0; k < j; k++) {
          w -= A->at(i, k) * A->at(k, j);
        }
        A->at(i, j) = w / A->at(j, j);
      }
      for (int j = i; j < n; j++) {
        w = A->at(i, j);
        for (int k = 0; k < i; k++) {
          w -= A->at(i, k) * A->at(k, j);
        }
        A->at(i, j) = w;
      }
    }
    for (int i = 0; i < n; i++) {
      w = b->at(i);
      for (int j = 0; j < i; j++)
        w -= A->at(i, j) * y->at(j);
      y->at(i) = w;
    }
    for (int i = n - 1; i >= 0; i--) {
      w = y->at(i);
      for (int j = i + 1; j < n; j++)
        w -= A->at(i, j) * x->at(j);
      x->at(i) = w / A->at(i, i);
    }
  }
};
} // OpenMP
} // CPlusPlus
#endif
