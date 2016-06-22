#ifndef _BASE_DURBIN_HPP_
#define _BASE_DURBIN_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class durbin : public PolyBenchKernel {
public:
  Arr1D<T> *r, *y;
  int n;

  durbin(std::string name, int n_) : PolyBenchKernel{name}, n{n_} {
    r = new Arr1D<T>{n};
    y = new Arr1D<T>{n};
  }

  ~durbin() {
    delete r;
    delete y;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr1D<T>::compare(
      this->y, dynamic_cast<const durbin *>(other)->y, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
