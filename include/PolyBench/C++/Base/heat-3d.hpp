#ifndef _CPP_BASE_HEAT_3D_HPP_
#define _CPP_BASE_HEAT_3D_HPP_

#include "PolyBench/Base/heat-3d.hpp"

namespace CPlusPlus {
namespace Base {
template <typename T>
class heat_3d : public ::Base::heat_3d<T> {
  using Parent = ::Base::heat_3d<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  heat_3d(Args... args) : ::Base::heat_3d<T>{"HEAT-3D - C++ Base", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, A, B);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
          A->at(i, j, k) = B->at(i, j, k) =
            static_cast<T>(i + j + (n - k)) * 10 / (n);
  }

  virtual void exec() {
    USE(READ, n, tsteps);
    USE(READWRITE, A, B);
    for (int t = 1; t <= tsteps; t++) {
      for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
          for (int k = 1; k < n - 1; k++) {
            B->at(i, j, k) =
              static_cast<T>(0.125) * (A->at(i + 1, j, k) - 2.0 * A->at(i, j, k)
                                       + A->at(i - 1, j, k))
              + static_cast<T>(0.125)
                  * (A->at(i, j + 1, k) - 2.0 * A->at(i, j, k)
                     + A->at(i, j - 1, k))
              + static_cast<T>(0.125)
                  * (A->at(i, j, k + 1) - 2.0 * A->at(i, j, k)
                     + A->at(i, j, k - 1))
              + A->at(i, j, k);
          }
        }
      }
      for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
          for (int k = 1; k < n - 1; k++) {
            A->at(i, j, k) =
              static_cast<T>(0.125) * (B->at(i + 1, j, k) - 2.0 * B->at(i, j, k)
                                       + B->at(i - 1, j, k))
              + static_cast<T>(0.125)
                  * (B->at(i, j + 1, k) - 2.0 * B->at(i, j, k)
                     + B->at(i, j - 1, k))
              + static_cast<T>(0.125)
                  * (B->at(i, j, k + 1) - 2.0 * B->at(i, j, k)
                     + B->at(i, j, k - 1))
              + B->at(i, j, k);
          }
        }
      }
    }
  }
};
} // Base
} // CPlusPlus
#endif
