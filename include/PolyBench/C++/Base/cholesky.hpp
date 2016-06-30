#ifndef _CPP_BASE_CHOLESKY_HPP_
#define _CPP_BASE_CHOLESKY_HPP_

#include "PolyBench/Base/cholesky.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class cholesky : public ::Base::cholesky<T>
{
  using Parent = ::Base::cholesky<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  cholesky(Args... args)
      : ::Base::cholesky<T>{std::string{"CHOLESKY - C++ Base"}, args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A);
    Arr2D<T> *B = new Arr2D<T>{n, n};
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++)
        A->at(i, j) = static_cast<T>(-j % n) / n + static_cast<T>(1.0);
      for (int j = i + 1; j < n; j++) {
        A->at(i, j) = static_cast<T>(0.0);
      }
      A->at(i, i) = static_cast<T>(1.0);
    }
    for (int r = 0; r < n; ++r)
      for (int s = 0; s < n; ++s)
        B->at(r, s) = static_cast<T>(0.0);
    for (int t = 0; t < n; ++t)
      for (int r = 0; r < n; ++r)
        for (int s = 0; s < n; ++s)
          B->at(r, s) += A->at(r, t) * A->at(s, t);
    for (int r = 0; r < n; ++r)
      for (int s = 0; s < n; ++s)
        A->at(r, s) = B->at(r, s);
    delete B;
  }

  virtual void exec()
  {
    USE(READ, n);
    USE(READWRITE, A);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        for (int k = 0; k < j; k++) {
          A->at(i, j) -= A->at(i, k) * A->at(j, k);
        }
        A->at(i, j) /= A->at(j, j);
      }
      for (int k = 0; k < i; k++) {
        A->at(i, i) -= A->at(i, k) * A->at(i, k);
      }
      A->at(i, i) = sqrt(A->at(i, i));
    }
  }
};
} // Base
} // CPlusPlus
#endif
