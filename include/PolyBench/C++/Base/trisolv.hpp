#ifndef _CPP_BASE_TRISOLV_HPP_
#define _CPP_BASE_TRISOLV_HPP_

#include "PolyBench/Base/trisolv.hpp"

namespace CPlusPlus {
namespace Base {
template <typename T>
class trisolv : public ::Base::trisolv<T> {
  using Parent = ::Base::trisolv<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  trisolv(Args... args) : ::Base::trisolv<T>{"TRISOLV - C++ Base", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, x, b, L);
    for (int i = 0; i < n; i++) {
      x->at(i) = -999;
      b->at(i) = i;
      for (int j = 0; j <= i; j++)
        L->at(i, j) = static_cast<T>(i + n - j + 1) * 2 / n;
    }
  }

  virtual void exec() {
    USE(READ, n, b, L);
    USE(READWRITE, x);
    for (int i = 0; i < n; i++) {
      x->at(i) = b->at(i);
      for (int j = 0; j < i; j++)
        x->at(i) -= L->at(i, j) * x->at(j);
      x->at(i) = x->at(i) / L->at(i, i);
    }
  }
};
} // Base
} // CPlusPlus
#endif
