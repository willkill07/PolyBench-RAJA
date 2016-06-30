#ifndef _CPP_BASE_GEMVER_HPP_
#define _CPP_BASE_GEMVER_HPP_

#include "PolyBench/Base/gemver.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class gemver : public ::Base::gemver<T>
{
  using Parent = ::Base::gemver<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gemver(Args... args) : ::Base::gemver<T>{"GEMVER - C++ Base", args...}
  {
  }

  virtual void init()
  {
    USE(READ, fn, n);
    USE(READWRITE, u1, u2, v1, v2, w, x, y, z, A);
    for (int i = 0; i < n; i++) {
      u1->at(i) = i;
      u2->at(i) = ((i + 1) / fn) / 2.0;
      v1->at(i) = ((i + 1) / fn) / 4.0;
      v2->at(i) = ((i + 1) / fn) / 6.0;
      y->at(i) = ((i + 1) / fn) / 8.0;
      z->at(i) = ((i + 1) / fn) / 9.0;
      x->at(i) = static_cast<T>(0.0);
      w->at(i) = static_cast<T>(0.0);
      for (int j = 0; j < n; j++)
        A->at(i, j) = static_cast<T>(i * j % n) / n;
    }
  }

  virtual void exec()
  {
    USE(READ, n, alpha, beta, u1, u2, v1, v2, y, z, A);
    USE(READWRITE, x, w);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        A->at(i, j) =
          A->at(i, j) + u1->at(i) * v1->at(j) + u2->at(i) * v2->at(j);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        x->at(i) = x->at(i) + beta * A->at(j, i) * y->at(j);
    for (int i = 0; i < n; i++)
      x->at(i) = x->at(i) + z->at(i);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        w->at(i) = w->at(i) + alpha * A->at(i, j) * x->at(j);
  }
};
} // Base
} // CPlusPlus
#endif
