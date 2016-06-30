#ifndef _CPP_OMP_JACOBI_2D_HPP_
#define _CPP_OMP_JACOBI_2D_HPP_

#include "PolyBench/Base/jacobi-2d.hpp"

namespace CPlusPlus
{
namespace OpenMP
{
template <typename T>
class jacobi_2d : public ::Base::jacobi_2d<T>
{
  using Parent = ::Base::jacobi_2d<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  jacobi_2d(Args... args)
      : ::Base::jacobi_2d<T>{"JACOBI-2D - C++ OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A, B);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        A->at(i, j) = (static_cast<T>(i) * (j + 2) + 2) / n;
        B->at(i, j) = (static_cast<T>(i) * (j + 3) + 3) / n;
      }
  }

  virtual void exec()
  {
    USE(READ, n, tsteps);
    USE(READWRITE, A, B);
    for (int t = 0; t < tsteps; t++) {
      for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < n - 1; j++)
          B->at(i, j) =
            static_cast<T>(0.2)
            * (A->at(i, j) + A->at(i, j - 1) + A->at(i, 1 + j) + A->at(1 + i, j)
               + A->at(i - 1, j));
      for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < n - 1; j++)
          A->at(i, j) =
            static_cast<T>(0.2)
            * (B->at(i, j) + B->at(i, j - 1) + B->at(i, 1 + j) + B->at(1 + i, j)
               + B->at(i - 1, j));
    }
  }
};
} // OpenMP
} // CPlusPlus
#endif
