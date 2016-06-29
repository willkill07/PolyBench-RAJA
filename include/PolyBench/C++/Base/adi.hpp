#ifndef _CPP_BASE_ADI_HPP_
#define _CPP_BASE_ADI_HPP_

#include "PolyBench/Base/adi.hpp"

namespace CPlusPlus {
namespace Base {
template <typename T>
class adi : public ::Base::adi<T> {
  using Parent = ::Base::adi<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  adi(Args... args) : ::Base::adi<T>{std::string{"ADI - C++ Base"}, args...} {
  }

  virtual void init() {
    USE(READWRITE, u);
    USE(READ, n);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        u->at(i, j) = static_cast<T>(i + n - j) / n;
  }
  virtual void exec() {
    USE(READ, n, tsteps);
    T DX(static_cast<T>(1.0) / n);
    T DY(static_cast<T>(1.0) / n);
    T DT(static_cast<T>(1.0) / tsteps);
    T B1(static_cast<T>(2.0));
    T B2(static_cast<T>(1.0));
    T mul1(B1 * DT / (DX * DX));
    T mul2(B2 * DT / (DY * DY));
    T a(-mul1 / static_cast<T>(2.0));
    T b(static_cast<T>(1.0) + mul1);
    T c(a);
    T d(-mul2 / static_cast<T>(2.0));
    T e(static_cast<T>(1.0) + mul2);
    T f(d);
    USE(READWRITE, v, u, p, q);
    for (int t = 1; t <= tsteps; t++) {
      for (int i = 1; i < n - 1; i++) {
        v->at(0, i) = static_cast<T>(1.0);
        p->at(i, 0) = static_cast<T>(0.0);
        q->at(i, 0) = v->at(0, i);
        for (int j = 1; j < n - 1; j++) {
          p->at(i, j) = -c / (a * p->at(i, j - 1) + b);
          q->at(i, j) = (-d * u->at(j, i - 1) + (1.0 + 2.0 * d) * u->at(j, i)
                         - f * u->at(j, i + 1)
                         - a * q->at(i, j - 1))
                        / (a * p->at(i, j - 1) + b);
        }
        v->at(n - 1, i) = static_cast<T>(1.0);
        for (int j = n - 2; j >= 1; j--) {
          v->at(j, i) = p->at(i, j) * v->at(j + 1, i) + q->at(i, j);
        }
      }
      for (int i = 1; i < n - 1; i++) {
        u->at(i, 0) = static_cast<T>(1.0);
        p->at(i, 0) = static_cast<T>(0.0);
        q->at(i, 0) = u->at(i, 0);
        for (int j = 1; j < n - 1; j++) {
          p->at(i, j) = -f / (d * p->at(i, j - 1) + e);
          q->at(i, j) = (-a * v->at(i - 1, j) + (1.0 + 2.0 * a) * v->at(i, j)
                         - c * v->at(i + 1, j)
                         - d * q->at(i, j - 1))
                        / (d * p->at(i, j - 1) + e);
        }
        u->at(i, n - 1) = static_cast<T>(1.0);
        for (int j = n - 2; j >= 1; j--) {
          u->at(i, j) = p->at(i, j) * u->at(i, j + 1) + q->at(i, j);
        }
      }
    }
  }
};
} // Base
} // CPlusPlus
#endif
