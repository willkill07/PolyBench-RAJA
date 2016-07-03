#ifndef _CPP_OMP_SEIDEL_2D_HPP_
#define _CPP_OMP_SEIDEL_2D_HPP_

#include "PolyBench/Base/seidel-2d.hpp"

namespace CPlusPlus
{
namespace OpenMP
{
template <typename T>
class seidel_2d : public ::Base::seidel_2d<T>
{
  using Parent = ::Base::seidel_2d<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  seidel_2d(Args... args)
  : ::Base::seidel_2d<T>{"SEIDEL-2D - C++ OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        A->at(i, j) = (static_cast<T>(i) * (j + 2) + 2) / n;
  }

  virtual void exec()
  {
    USE(READ, n, tsteps);
    USE(READWRITE, A);
    for (int t = 0; t < tsteps; t++)
      for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < n - 1; j++)
          A->at(i, j) =
            (A->at(i - 1, j - 1) + A->at(i - 1, j) + A->at(i - 1, j + 1)
             + A->at(i, j - 1)
             + A->at(i, j)
             + A->at(i, j + 1)
             + A->at(i + 1, j - 1)
             + A->at(i + 1, j)
             + A->at(i + 1, j + 1))
            / 9.0;
  }
};
} // OpenMP
} // CPlusPlus
#endif
