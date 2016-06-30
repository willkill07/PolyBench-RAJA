#ifndef _CPP_BASE_SYRK_HPP_
#define _CPP_BASE_SYRK_HPP_

#include "PolyBench/Base/syrk.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class syrk : public ::Base::syrk<T>
{
  using Parent = ::Base::syrk<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  syrk(Args... args) : ::Base::syrk<T>{"SYRK - C++ Base", args...}
  {
  }

  virtual void init()
  {
    USE(READ, m, n);
    USE(READWRITE, A, C);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++)
        A->at(i, j) = static_cast<T>((i * j + 1) % n) / n;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        C->at(i, j) = static_cast<T>((i * j + 2) % m) / m;
  }

  virtual void exec()
  {
    USE(READ, m, n, alpha, beta, A);
    USE(READWRITE, C);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++)
        C->at(i, j) *= beta;
      for (int k = 0; k < m; k++) {
        for (int j = 0; j <= i; j++)
          C->at(i, j) += alpha * A->at(i, k) * A->at(j, k);
      }
    }
  }
};
} // Base
} // CPlusPlus
#endif
