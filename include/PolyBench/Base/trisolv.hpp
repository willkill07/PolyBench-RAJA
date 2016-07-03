#ifndef _BASE_TRISOLV_HPP_
#define _BASE_TRISOLV_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class trisolv : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  std::shared_ptr<Arr2D<T>> L;
  std::shared_ptr<Arr1D<T>> x, b;

  trisolv(std::string name, int n_)
  : PolyBenchKernel{name},
    n{n_},
    L{new Arr2D<T>{n, n}},
    x{new Arr1D<T>{n}},
    b{new Arr1D<T>{n}}
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
