#ifndef _BASE_HEAT_3D_HPP_
#define _BASE_HEAT_3D_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class heat_3d : public PolyBenchKernel {
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n, tsteps;
  Arr3D<T> *A, *B;

  heat_3d(std::string name, int n_, int tsteps_)
      : PolyBenchKernel{name}, n{n_}, tsteps{tsteps_} {
    A = new Arr3D<T>{n, n, n};
    B = new Arr3D<T>{n, n, n};
  }

  ~heat_3d() {
    delete A;
    delete B;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr3D<T>::compare(
      this->A, dynamic_cast<const heat_3d *>(other)->A, static_cast<T>(0));
  }
};
} // Base
#endif
