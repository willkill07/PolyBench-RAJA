#ifndef _CPP_BASE_GESUMMV_HPP_
#define _CPP_BASE_GESUMMV_HPP_

#include "PolyBench/Base/gesummv.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class gesummv : public ::Base::gesummv<T>
{
  using Parent = ::Base::gesummv<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gesummv(Args... args) : ::Base::gesummv<T>{"GESUMMV - C++ Base", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A, B, x);
    for (int i = 0; i < n; i++) {
      x->at(i) = static_cast<T>(i % n) / n;
      for (int j = 0; j < n; j++) {
        A->at(i, j) = static_cast<T>((i * j + 1) % n) / n;
        B->at(i, j) = static_cast<T>((i * j + 2) % n) / n;
      }
    }
  }

  virtual void exec()
  {
    USE(READ, n, alpha, beta, A, B, x);
    USE(READWRITE, y, tmp);
    for (int i = 0; i < n; i++) {
      tmp->at(i) = static_cast<T>(0.0);
      y->at(i) = static_cast<T>(0.0);
      for (int j = 0; j < n; j++) {
        tmp->at(i) = A->at(i, j) * x->at(j) + tmp->at(i);
        y->at(i) = B->at(i, j) * x->at(j) + y->at(i);
      }
      y->at(i) = alpha * tmp->at(i) + beta * y->at(i);
    }
  }
};
} // Base
} // CPlusPlus
#endif
