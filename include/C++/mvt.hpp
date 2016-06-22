#ifndef _CPP_MVT_HPP_
#define _CPP_MVT_HPP_

#include "Base/mvt.hpp"

namespace CPlusPlus {
template <typename T>
class mvt : public Base::mvt<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 1>::type>
  mvt(Args... args) : Base::mvt<T>{"MVT - Vanilla", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A, x1, x2, y_1, y_2);
    for (int i = 0; i < n; i++) {
      x1->at(i) = static_cast<T>(i % n) / n;
      x2->at(i) = static_cast<T>((i + 1) % n) / n;
      y_1->at(i) = static_cast<T>((i + 3) % n) / n;
      y_2->at(i) = static_cast<T>((i + 4) % n) / n;
      for (int j = 0; j < n; j++)
        A->at(i, j) = static_cast<T>(i * j % n) / n;
    }
  }

  virtual void exec() {
    USE(READ, n, A, y_1, y_2);
    USE(READWRITE, x1, x2);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        x1->at(i) = x1->at(i) + A->at(i, j) * y_1->at(j);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        x2->at(i) = x2->at(i) + A->at(j, i) * y_2->at(j);
  }
};
} // CPlusPlus
#endif
