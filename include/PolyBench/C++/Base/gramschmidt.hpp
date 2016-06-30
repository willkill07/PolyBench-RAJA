#ifndef _CPP_BASE_GRAMSCHMIDT_HPP_
#define _CPP_BASE_GRAMSCHMIDT_HPP_

#include "PolyBench/Base/gramschmidt.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class gramschmidt : public ::Base::gramschmidt<T>
{
  using Parent = ::Base::gramschmidt<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gramschmidt(Args... args)
      : ::Base::gramschmidt<T>{"GRAMSCHMIDT - C++ Base", args...}
  {
  }

  virtual void init()
  {
    USE(READ, m, n);
    USE(READWRITE, A, R, Q);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        A->at(i, j) = ((static_cast<T>((i * j) % m) / m) * 100) + 10;
        Q->at(i, j) = static_cast<T>(0.0);
      }
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        R->at(i, j) = static_cast<T>(0.0);
  }

  virtual void exec()
  {
    USE(READ, m, n);
    USE(READWRITE, A, R, Q);
    for (int k = 0; k < n; k++) {
      T nrm = static_cast<T>(0.0);
      for (int i = 0; i < m; i++)
        nrm += A->at(i, k) * A->at(i, k);
      R->at(k, k) = sqrt(nrm);
      for (int i = 0; i < m; i++)
        Q->at(i, k) = A->at(i, k) / R->at(k, k);
      for (int j = k + 1; j < n; j++) {
        R->at(k, j) = static_cast<T>(0.0);
        for (int i = 0; i < m; i++)
          R->at(k, j) += Q->at(i, k) * A->at(i, j);
        for (int i = 0; i < m; i++)
          A->at(i, j) = A->at(i, j) - Q->at(i, k) * R->at(k, j);
      }
    }
  }
};
} // Base
} // CPlusPlus
#endif
