#ifndef _BASE_LUDCMP_HPP_
#define _BASE_LUDCMP_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class ludcmp : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  T fn;
  std::shared_ptr<Arr2D<T>> A;
  std::shared_ptr<Arr1D<T>> b, x, y;

  ludcmp(std::string name, int n_)
  : PolyBenchKernel{name},
    n{n_},
    fn{static_cast<T>(n)},
    A{new Arr2D<T>{n, n}},
    b{new Arr1D<T>{n}},
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
    return o && Arr1D<T>::compare(this->x.get(), o->x.get(), eps);
  }
};
} // Base
#endif
