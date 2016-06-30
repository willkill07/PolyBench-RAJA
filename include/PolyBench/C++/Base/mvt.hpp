#ifndef _CPP_BASE_MVT_HPP_
#define _CPP_BASE_MVT_HPP_

#include "PolyBench/Base/mvt.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class mvt : public ::Base::mvt<T>
{
  using Parent = ::Base::mvt<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  mvt(Args... args) : ::Base::mvt<T>{"MVT - C++ Base", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A, x1, x2, y_1, y_2);
    for (int i = 0; i < n; i++) {
      x1->at(i) = static_cast<T>(i % n) / n;
      x2->at(i) = static_cast<T>((i + 1) % n) / n;
      y_1->at(i) = static_cast<T>((i + 3) % n) / n;
      y_2->at(i) = static_cast<T>((i + 4) % n) / n;
      for (int j = 0; j < n; j++)
        A->at(i, j) = static_cast<T>(i * j % n) / n;
    }
  }

  virtual void exec()
  {
    USE(READ, n, A, y_1, y_2);
    USE(READWRITE, x1, x2);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        x1->at(i) = x1->at(i) + A->at(i, j) * y_1->at(j);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        x2->at(i) = x2->at(i) + A->at(j, i) * y_2->at(j);
  }
};
} // Base
} // CPlusPlus
#endif
