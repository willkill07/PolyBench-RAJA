#ifndef _CPP_ATAX_HPP_
#define _CPP_ATAX_HPP_

#include "Base/atax.hpp"

namespace CPlusPlus {
template <typename T>
class atax : public Base::atax<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  atax(Args... args) : Base::atax<T>{std::string{"ATAX - Vanilla"}, args...} {
  }

  virtual void init() {
    USE(READ, n, m);
    USE(READWRITE, A, x);
    T fn{static_cast<T>(n)};
    for (int i = 0; i < n; i++)
      x->at(i) = 1 + (i / fn);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        A->at(i, j) = static_cast<T>((i + j) % n) / (5 * m);
  }

  virtual void exec() {
    USE(READ, n, m, A, x);
    USE(READWRITE, y, tmp);
    for (int i = 0; i < n; i++)
      y->at(i) = static_cast<T>(0.0);
    for (int i = 0; i < m; i++) {
      tmp->at(i) = static_cast<T>(0.0);
      for (int j = 0; j < n; j++)
        tmp->at(i) = tmp->at(i) + A->at(i, j) * x->at(j);
      for (int j = 0; j < n; j++)
        y->at(j) = y->at(j) + A->at(i, j) * tmp->at(i);
    }
  }
};
} // CPlusPlus
#endif
