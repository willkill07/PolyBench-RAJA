#ifndef _BASE_GESUMMV_HPP_
#define _BASE_GESUMMV_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class gesummv : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  T alpha, beta;
  std::shared_ptr<Arr2D<T>> A, B;
  std::shared_ptr<Arr1D<T>> tmp, x, y;

  gesummv(std::string name, int n_)
  : PolyBenchKernel{name},
    n{n_},
    alpha{static_cast<T>(1.5)},
    beta{static_cast<T>(1.2)},
    A{new Arr2D<T>{n, n}},
    B{new Arr2D<T>{n, n}},
    tmp{new Arr1D<T>{n}},
    x{new Arr1D<T>{n}},
    y{new Arr1D<T>{n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr1D<T>::compare(this->y.get(), o->y.get(), eps);
  }
};
} // Base
#endif
