#ifndef _CPP_HEAT_3D_HPP_
#define _CPP_HEAT_3D_HPP_

#include "Base/heat-3d.hpp"

namespace CPlusPlus {
template <typename T>
class heat_3d : public Base::heat_3d<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  heat_3d(Args... args) : Base::heat_3d<T>{"HEAT-3D - Vanilla", args...} {
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
} // CPlusPlus
#endif
