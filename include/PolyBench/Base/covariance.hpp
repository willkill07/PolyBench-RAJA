#ifndef _BASE_COVARIANCE_HPP_
#define _BASE_COVARIANCE_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class covariance : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n, m;
  T float_n;
  Arr2D<T> *data, *cov;
  Arr1D<T> *mean;

  covariance(std::string name, int n_, int m_)
      : PolyBenchKernel{name}, n{n_}, m{m_}, float_n{static_cast<T>(n)}
  {
    data = new Arr2D<T>{n, m};
    cov = new Arr2D<T>{m, m};
    mean = new Arr1D<T>{m};
  }

  ~covariance()
  {
    delete data;
    delete cov;
    delete mean;
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr2D<T>::compare(
      this->cov,
      dynamic_cast<const covariance *>(other)->cov,
      static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
