#ifndef _CPP_OMP_FDTD_2D_HPP_
#define _CPP_OMP_FDTD_2D_HPP_

#include "PolyBench/Base/fdtd-2d.hpp"

namespace CPlusPlus {
namespace OpenMP {
template <typename T>
class fdtd_2d : public ::Base::fdtd_2d<T> {
public:
  template <typename... Args>
  fdtd_2d(Args... args) : ::Base::fdtd_2d<T>{"FDTD-2D - C++ OpenMP", args...} {
  }

  virtual void init() {
    USE(READ, tmax, nx, ny);
    USE(READWRITE, _fict_, ex, ey, hz);
    for (int i = 0; i < tmax; i++)
      _fict_->at(i) = static_cast<T>(i);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++) {
        ex->at(i, j) = (static_cast<T>(i) * (j + 1)) / nx;
        ey->at(i, j) = (static_cast<T>(i) * (j + 2)) / ny;
        hz->at(i, j) = (static_cast<T>(i) * (j + 3)) / nx;
      }
  }

  virtual void exec() {
    USE(READ, nx, ny, _fict_, tmax);
    USE(READWRITE, ex, ey, hz);
    for (int t = 0; t < tmax; t++) {
      for (int j = 0; j < ny; j++)
        ey->at(0, j) = _fict_->at(t);
      for (int i = 1; i < nx; i++)
        for (int j = 0; j < ny; j++)
          ey->at(i, j) = ey->at(i, j) - 0.5 * (hz->at(i, j) - hz->at(i - 1, j));
      for (int i = 0; i < nx; i++)
        for (int j = 1; j < ny; j++)
          ex->at(i, j) = ex->at(i, j) - 0.5 * (hz->at(i, j) - hz->at(i, j - 1));
      for (int i = 0; i < nx - 1; i++)
        for (int j = 0; j < ny - 1; j++)
          hz->at(i, j) =
            hz->at(i, j)
            - 0.7 * (ex->at(i, j + 1) - ex->at(i, j) + ey->at(i + 1, j)
                     - ey->at(i, j));
    }
  }
};
} // OpenMP
} // CPlusPlus
#endif
