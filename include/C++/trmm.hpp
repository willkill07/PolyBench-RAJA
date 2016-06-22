#ifndef _CPP_TRMM_HPP_
#define _CPP_TRMM_HPP_

#include "Base/trmm.hpp"

namespace CPlusPlus {
template <typename T>
class trmm : public Base::trmm<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  trmm(Args... args) : Base::trmm<T>{"TRMM - Vanilla", args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, A, B);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < i; j++) {
        A->at(i, j) = static_cast<T>((i + j) % m) / m;
      }
      A->at(i, i) = static_cast<T>(1.0);
      for (int j = 0; j < n; j++) {
        B->at(i, j) = static_cast<T>((n + (i - j)) % n) / n;
      }
    }
  }

  virtual void exec() {
    USE(READ, m, n, alpha, A);
    USE(READWRITE, B);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        for (int k = i + 1; k < m; k++)
          B->at(i, j) += A->at(k, i) * B->at(k, j);
        B->at(i, j) = alpha * B->at(i, j);
      }
  }
};
} // CPlusPlus
#endif
