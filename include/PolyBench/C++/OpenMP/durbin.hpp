#ifndef _CPP_OMP_DURBIN_HPP_
#define _CPP_OMP_DURBIN_HPP_

#include "PolyBench/Base/durbin.hpp"

namespace CPlusPlus
{
namespace OpenMP
{
template <typename T>
class durbin : public ::Base::durbin<T>
{
  using Parent = ::Base::durbin<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  durbin(Args... args) : ::Base::durbin<T>{"DURBIN - C++ OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, r);
    for (int i = 0; i < n; i++) {
      r->at(i) = (n + 1 - i);
    }
  }

  virtual void exec()
  {
    USE(READ, n, r);
    USE(READWRITE, y);
    Arr1D<T> *z = new Arr1D<T>{n};
    y->at(0) = -r->at(0);
    T beta = static_cast<T>(1.0);
    T alpha = -r->at(0);
    for (int k = 1; k < n; k++) {
      beta = (1 - alpha * alpha) * beta;
      T sum = static_cast<T>(0.0);
      for (int i = 0; i < k; i++) {
        sum += r->at(k - i - 1) * y->at(i);
      }
      alpha = -(r->at(k) + sum) / beta;
      for (int i = 0; i < k; i++) {
        z->at(i) = y->at(i) + alpha * y->at(k - i - 1);
      }
      for (int i = 0; i < k; i++) {
        y->at(i) = z->at(i);
      }
      y->at(k) = alpha;
    }
    delete z;
  }
};
} // OpenMP
} // CPlusPlus
#endif
