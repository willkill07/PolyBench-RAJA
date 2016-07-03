#ifndef _CPP_BASE_JACOBI_1D_HPP_
#define _CPP_BASE_JACOBI_1D_HPP_

#include "PolyBench/Base/jacobi-1d.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class jacobi_1d : public ::Base::jacobi_1d<T>
{
  using Parent = ::Base::jacobi_1d<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  jacobi_1d(Args... args)
  : ::Base::jacobi_1d<T>{"JACOBI-1D - C++ Base", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A, B);
    for (int i = 0; i < n; i++) {
      A->at(i) = (static_cast<T>(i) + 2) / n;
      B->at(i) = (static_cast<T>(i) + 3) / n;
    }
  }

  virtual void exec()
  {
    USE(READ, n, tsteps);
    USE(READWRITE, A, B);
    for (int t = 0; t < tsteps; t++) {
      for (int i = 1; i < n - 1; i++)
        B->at(i) =
          static_cast<T>(0.33333) * (A->at(i - 1) + A->at(i) + A->at(i + 1));
      for (int i = 1; i < n - 1; i++)
        A->at(i) =
          static_cast<T>(0.33333) * (B->at(i - 1) + B->at(i) + B->at(i + 1));
    }
  }
};
} // Base
} // CPlusPlus
#endif
