#ifndef _BASE_ATAX_HPP_
#define _BASE_ATAX_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class atax : public PolyBenchKernel {
public:
  Arr2D<T> *A;
  Arr1D<T> *x, *y, *tmp;
  int m, n;

  atax(std::string name, int m_, int n_) : PolyBenchKernel{name}, m{m_}, n{n_} {
    A = new Arr2D<T>{m, n};
    x = new Arr1D<T>{n};
    y = new Arr1D<T>{n};
    tmp = new Arr1D<T>{m};
  }

  ~atax() {
    delete A;
    delete x;
    delete y;
    delete tmp;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr1D<T>::compare(
      this->y, dynamic_cast<const atax *>(other)->y, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
