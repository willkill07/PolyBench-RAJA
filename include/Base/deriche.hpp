#ifndef _BASE_DERICHE_HPP_
#define _BASE_DERICHE_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class deriche : public PolyBenchKernel {
public:
  Arr2D<T> *imgIn, *imgOut, *y1, *y2;
  int w, h;
  T alpha;

  deriche(std::string name, int w_, int h_)
      : PolyBenchKernel{name}, w{w_}, h{h_}, alpha{static_cast<T>(0.25)} {
    imgIn = new Arr2D<T>{w, h};
    imgOut = new Arr2D<T>{w, h};
    y1 = new Arr2D<T>{w, h};
    y2 = new Arr2D<T>{w, h};
  }

  ~deriche() {
    delete imgIn;
    delete imgOut;
    delete y1;
    delete y2;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->imgOut,
      dynamic_cast<const deriche *>(other)->imgOut,
      static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
