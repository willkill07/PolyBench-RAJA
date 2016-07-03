#ifndef _BASE_MVT_HPP_
#define _BASE_MVT_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class mvt : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  std::shared_ptr<Arr2D<T>> A;
  std::shared_ptr<Arr1D<T>> x1, x2, y_1, y_2;

  mvt(std::string name, int n_)
  : PolyBenchKernel{name},
    n{n_},
    A{new Arr2D<T>{n, n}},
    x1{new Arr1D<T>{n}},
    x2{new Arr1D<T>{n}},
    y_1{new Arr1D<T>{n}},
    y_2{new Arr1D<T>{n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr1D<T>::compare(this->x1.get(), o->x1.get(), eps)
           && Arr1D<T>::compare(this->x2.get(), o->x2.get(), eps);
  }
};
} // Base
#endif
