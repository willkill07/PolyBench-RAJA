#ifndef _BASE_2MM_HPP_
#define _BASE_2MM_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class mm2 : public PolyBenchKernel {
public:
  T alpha, beta;
  Arr2D<T> *A, *B, *C, *D, *tmp;
  int ni, nj, nk, nl;

  mm2(std::string n, int ni_, int nj_, int nk_, int nl_)
      : PolyBenchKernel{n}, ni{ni_}, nj{nj_}, nk{nk_}, nl{nl_} {
    A = new Arr2D<T>{ni, nk};
    B = new Arr2D<T>{nk, nj};
    C = new Arr2D<T>{nj, nl};
    D = new Arr2D<T>{ni, nl};
    tmp = new Arr2D<T>{ni, nj};
  }

  ~mm2() {
    delete A;
    delete B;
    delete C;
    delete D;
    delete tmp;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->D, dynamic_cast<const mm2 *>(other)->D, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
