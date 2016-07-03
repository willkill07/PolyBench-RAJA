#ifndef _BASE_GEMVER_HPP_
#define _BASE_GEMVER_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class gemver : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  T alpha, beta, fn;
  std::shared_ptr<Arr2D<T>> A;
  std::shared_ptr<Arr1D<T>> u1, u2, v1, v2, w, x, y, z;

  gemver(std::string name, int n_)
  : PolyBenchKernel{name},
    n{n_},
    alpha{static_cast<T>(1.5)},
    beta{static_cast<T>(1.2)},
    fn{static_cast<T>(n)},
    A{new Arr2D<T>{n, n}},
    u1{new Arr1D<T>{n}},
    u2{new Arr1D<T>{n}},
    v1{new Arr1D<T>{n}},
    v2{new Arr1D<T>{n}},
    w{new Arr1D<T>{n}},
    x{new Arr1D<T>{n}},
    y{new Arr1D<T>{n}},
    z{new Arr1D<T>{n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr1D<T>::compare(this->w.get(), o->w.get(), eps);
  }
};
} // Base
#endif
