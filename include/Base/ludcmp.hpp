#ifndef _BASE_LUDCMP_HPP_
#define _BASE_LUDCMP_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class ludcmp : public PolyBenchKernel {
public:
  int n;
  T fn;
  Arr2D<T> *A;
  Arr1D<T> *b, *x, *y;

  ludcmp(std::string name, int n_)
      : PolyBenchKernel{name}, n{n_}, fn{static_cast<T>(n)} {
    A = new Arr2D<T>{n, n};
    b = new Arr1D<T>{n};
    x = new Arr1D<T>{n};
    y = new Arr1D<T>{n};
  }

  ~ludcmp() {
    delete A;
    delete b;
    delete x;
    delete y;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr1D<T>::compare(
      this->x, dynamic_cast<const ludcmp *>(other)->x, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
