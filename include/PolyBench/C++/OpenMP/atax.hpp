#ifndef _CPP_OMP_ATAX_HPP_
#define _CPP_OMP_ATAX_HPP_

#include "PolyBench/Base/atax.hpp"

namespace CPlusPlus
{
namespace OpenMP
{
template <typename T>
class atax : public ::Base::atax<T>
{
  using Parent = ::Base::atax<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  atax(Args... args)
      : ::Base::atax<T>{std::string{"ATAX - C++ OpenMP"}, args...}
  {
  }

  virtual void init()
  {
    USE(READ, n, m);
    USE(READWRITE, A, x);
    T fn{static_cast<T>(n)};
    for (int i = 0; i < n; i++)
      x->at(i) = 1 + (i / fn);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        A->at(i, j) = static_cast<T>((i + j) % n) / (5 * m);
  }

  virtual void exec()
  {
    USE(READ, n, m, A, x);
    USE(READWRITE, y, tmp);
    for (int i = 0; i < n; i++)
      y->at(i) = static_cast<T>(0.0);
    for (int i = 0; i < m; i++) {
      tmp->at(i) = static_cast<T>(0.0);
      for (int j = 0; j < n; j++)
        tmp->at(i) = tmp->at(i) + A->at(i, j) * x->at(j);
      for (int j = 0; j < n; j++)
        y->at(j) = y->at(j) + A->at(i, j) * tmp->at(i);
    }
  }
};
} // OpenMP
} // CPlusPlus
#endif
