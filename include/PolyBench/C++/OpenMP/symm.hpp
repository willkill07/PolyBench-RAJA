#ifndef _CPP_OMP_SYMM_HPP_
#define _CPP_OMP_SYMM_HPP_

#include "PolyBench/Base/symm.hpp"

namespace CPlusPlus
{
namespace OpenMP
{
template <typename T>
class symm : public ::Base::symm<T>
{
  using Parent = ::Base::symm<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  symm(Args... args) : ::Base::symm<T>{"SYMM - C++ OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, m, n);
    USE(READWRITE, A, C, B);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        C->at(i, j) = static_cast<T>((i + j) % 100) / m;
        B->at(i, j) = static_cast<T>((n + i - j) % 100) / m;
      }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j <= i; j++)
        A->at(i, j) = static_cast<T>((i + j) % 100) / m;
      for (int j = i + 1; j < m; j++)
        A->at(i, j) = -999;
    }
  }

  virtual void exec()
  {
    USE(READ, m, n, alpha, beta, A, B);
    USE(READWRITE, C);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        T temp2 = 0;
        for (int k = 0; k < i; k++) {
          C->at(k, j) += alpha * B->at(i, j) * A->at(i, k);
          temp2 += B->at(k, j) * A->at(i, k);
        }
        C->at(i, j) = beta * C->at(i, j) + alpha * B->at(i, j) * A->at(i, i)
                      + alpha * temp2;
      }
  }
};
} // OpenMP
} // CPlusPlus
#endif
