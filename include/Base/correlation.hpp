#ifndef _BASE_CORRELATION_HPP_
#define _BASE_CORRELATION_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class correlation : public PolyBenchKernel {
public:
  int n, m;
  T float_n;
  Arr2D<T> *data, *corr;
  Arr1D<T> *mean, *stddev;

  correlation(std::string name, int n_, int m_)
      : PolyBenchKernel{name}, n{n_}, m{m_}, float_n{static_cast<T>(n)} {
    data = new Arr2D<T>{n, m};
    corr = new Arr2D<T>{m, m};
    mean = new Arr1D<T>{m};
    stddev = new Arr1D<T>{m};
  }

  ~correlation() {
    delete data;
    delete corr;
    delete mean;
    delete stddev;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->corr,
      dynamic_cast<const correlation *>(other)->corr,
      static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
