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
  std::shared_ptr<Arr2D<T>> data, cov;
  std::shared_ptr<Arr1D<T>> mean;

  covariance(std::string name, int n_, int m_)
  : PolyBenchKernel{name},
    n{n_},
    m{m_},
    float_n{static_cast<T>(n)},
    data{new Arr2D<T>{n, m}},
    cov{new Arr2D<T>{m, m}},
    mean{new Arr1D<T>{m}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->cov.get(), o->cov.get(), eps);
  }
};
} // Base
#endif
