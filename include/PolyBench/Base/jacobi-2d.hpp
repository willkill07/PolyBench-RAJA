#ifndef _BASE_JACOBI_2D_HPP_
#define _BASE_JACOBI_2D_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class jacobi_2d : public PolyBenchKernel {
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n, tsteps;
  Arr2D<T> *A, *B;

  jacobi_2d(std::string name, int n_, int tsteps_)
      : PolyBenchKernel{name}, n{n_}, tsteps{tsteps_} {
    A = new Arr2D<T>{n, n};
    B = new Arr2D<T>{n, n};
  }

  ~jacobi_2d() {
    delete A;
    delete B;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->A, dynamic_cast<const jacobi_2d *>(other)->A, static_cast<T>(0));
  }
};
} // Base
#endif
