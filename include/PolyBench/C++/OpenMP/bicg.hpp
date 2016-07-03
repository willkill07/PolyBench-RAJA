#ifndef _CPP_OMP_BICG_HPP_
#define _CPP_OMP_BICG_HPP_

#include "PolyBench/Base/bicg.hpp"

namespace CPlusPlus
{
namespace OpenMP
{
template <typename T>
class bicg : public ::Base::bicg<T>
{
  using Parent = ::Base::bicg<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  bicg(Args... args)
  : ::Base::bicg<T>{std::string{"BICG - C++ OpenMP"}, args...}
  {
  }

  virtual void init()
  {
    USE(READ, m, n);
    USE(READWRITE, p, r, A);
    for (int i = 0; i < m; i++)
      p->at(i) = static_cast<T>(i % m) / m;
    for (int i = 0; i < n; i++) {
      r->at(i) = static_cast<T>(i % n) / n;
      for (int j = 0; j < m; j++)
        A->at(i, j) = static_cast<T>(i * (j + 1) % n) / n;
    }
  }

  virtual void exec()
  {
    USE(READ, m, n, A, r, p);
    USE(READWRITE, s, q);
    for (int i = 0; i < m; i++)
      s->at(i) = static_cast<T>(0.0);
    for (int i = 0; i < n; i++) {
      q->at(i) = static_cast<T>(0.0);
      for (int j = 0; j < m; j++) {
        s->at(j) = s->at(j) + r->at(i) * A->at(i, j);
        q->at(i) = q->at(i) + A->at(i, j) * p->at(j);
      }
    }
  }
};
} // OpenMP
} // CPlusPlus
#endif
