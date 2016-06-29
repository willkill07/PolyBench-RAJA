#ifndef _BASE_ADI_HPP_
#define _BASE_ADI_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class adi : public PolyBenchKernel {
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  Arr2D<T> *u, *v, *p, *q;
  int n, tsteps;

  adi(std::string name, int n_, int tsteps_)
      : PolyBenchKernel{name}, n{n_}, tsteps{tsteps_} {
    u = new Arr2D<T>{n, n};
    v = new Arr2D<T>{n, n};
    p = new Arr2D<T>{n, n};
    q = new Arr2D<T>{n, n};
  }

  ~adi() {
    delete u;
    delete v;
    delete p;
    delete q;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->u, dynamic_cast<const adi *>(other)->u, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
