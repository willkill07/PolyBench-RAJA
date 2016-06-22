#ifndef _CPP_FLOYD_WARSHALL_HPP_
#define _CPP_FLOYD_WARSHALL_HPP_

#include "Base/floyd-warshall.hpp"

namespace CPlusPlus {
template <typename T>
class floyd_warshall : public Base::floyd_warshall<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 1>::type>
  floyd_warshall(Args... args)
      : Base::floyd_warshall<T>{"FLOYD-WARSHALL - Vanilla", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, path);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        path->at(i, j) = i * j % 7 + 1;
        if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
          path->at(i, j) = 999;
      }
  }

  virtual void exec() {
    USE(READ, n);
    USE(READWRITE, path);
    for (int k = 0; k < n; k++) {
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
          path->at(i, j) = path->at(i, j) < path->at(i, k) + path->at(k, j)
                             ? path->at(i, j)
                             : path->at(i, k) + path->at(k, j);
    }
  }
};
} // CPlusPlus
#endif
