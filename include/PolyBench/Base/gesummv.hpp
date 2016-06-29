#ifndef _BASE_GESUMMV_HPP_
#define _BASE_GESUMMV_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class gesummv : public PolyBenchKernel {
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  T alpha{static_cast<T>(1.5)}, beta{static_cast<T>(1.2)};
  Arr2D<T> *A, *B;
  Arr1D<T> *tmp, *x, *y;

  gesummv(std::string name, int n_) : PolyBenchKernel{name}, n{n_} {
    A = new Arr2D<T>{n, n};
    B = new Arr2D<T>{n, n};
    tmp = new Arr1D<T>{n};
    x = new Arr1D<T>{n};
    y = new Arr1D<T>{n};
  }

  ~gesummv() {
    delete A;
    delete B;
    delete tmp;
    delete x;
    delete y;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr1D<T>::compare(
      this->y, dynamic_cast<const gesummv *>(other)->y, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
