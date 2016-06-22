#ifndef _BASE_SYR2K_HPP_
#define _BASE_SYR2K_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class syr2k : public PolyBenchKernel {
public:
  int m, n;
  T alpha{static_cast<T>(1.5)}, beta{static_cast<T>(1.2)};
  Arr2D<T> *A, *C, *B;

  syr2k(std::string name, int m_, int n_)
      : PolyBenchKernel{name}, m{m_}, n{n_} {
    A = new Arr2D<T>{n, m};
    C = new Arr2D<T>{n, n};
    B = new Arr2D<T>{n, m};
  }

  ~syr2k() {
    delete A;
    delete C;
    delete B;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->C, dynamic_cast<const syr2k *>(other)->C, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
