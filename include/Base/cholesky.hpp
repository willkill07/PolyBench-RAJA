#ifndef _BASE_CHOLESKY_HPP_
#define _BASE_CHOLESKY_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class cholesky : public PolyBenchKernel {
public:
  Arr2D<T> *A;
  int n;

  cholesky(std::string name, int n_) : PolyBenchKernel{name}, n{n_} {
    A = new Arr2D<T>{n, n};
  }
  ~cholesky() {
    delete A;
  }
  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->A,
      dynamic_cast<const cholesky *>(other)->A,
      static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
