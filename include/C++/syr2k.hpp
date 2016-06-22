#ifndef _CPP_SYR2K_HPP_
#define _CPP_SYR2K_HPP_

#include "Base/syr2k.hpp"

namespace CPlusPlus {
template <typename T>
class syr2k : public Base::syr2k<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  syr2k(Args... args) : Base::syr2k<T>{"SYR2K - Vanilla", args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, A, C, B);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) {
        A->at(i, j) = static_cast<T>((i * j + 1) % n) / n;
        B->at(i, j) = static_cast<T>((i * j + 2) % m) / m;
      }
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        C->at(i, j) = static_cast<T>((i * j + 3) % n) / m;
      }
  }

  virtual void exec() {
    USE(READ, m, n, alpha, beta, A, B);
    USE(READWRITE, C);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++)
        C->at(i, j) *= beta;
      for (int k = 0; k < m; k++)
        for (int j = 0; j <= i; j++) {
          C->at(i, j) += A->at(j, k) * alpha * B->at(i, k)
                         + B->at(j, k) * alpha * A->at(i, k);
        }
    }
  }
};
} // CPlusPlus
#endif
