#ifndef _BASE_JACOBI_1D_HPP_
#define _BASE_JACOBI_1D_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class jacobi_1d : public PolyBenchKernel {
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n, tsteps;
  Arr1D<T> *A, *B;

  jacobi_1d(std::string name, int n_, int tsteps_)
      : PolyBenchKernel{name}, n{n_}, tsteps{tsteps_} {
    A = new Arr1D<T>{n};
    B = new Arr1D<T>{n};
  }

  ~jacobi_1d() {
    delete A;
    delete B;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr1D<T>::compare(
      this->A, dynamic_cast<const jacobi_1d *>(other)->A, static_cast<T>(0));
  }
};
} // Base
#endif
