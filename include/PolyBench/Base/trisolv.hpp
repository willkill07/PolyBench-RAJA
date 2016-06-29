#ifndef _BASE_TRISOLV_HPP_
#define _BASE_TRISOLV_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class trisolv : public PolyBenchKernel {
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  Arr2D<T> *L;
  Arr1D<T> *x, *b;

  trisolv(std::string name, int n_) : PolyBenchKernel{name}, n{n_} {
    L = new Arr2D<T>{n, n};
    x = new Arr1D<T>{n};
    b = new Arr1D<T>{n};
  }

  ~trisolv() {
    delete L;
    delete x;
    delete b;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr1D<T>::compare(
      this->x, dynamic_cast<const trisolv *>(other)->x, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
