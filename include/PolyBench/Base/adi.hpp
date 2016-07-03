#ifndef _BASE_ADI_HPP_
#define _BASE_ADI_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class adi : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n, tsteps;
  std::shared_ptr<Arr2D<T>> u, v, p, q;

  adi(std::string name, int n_, int tsteps_)
  : PolyBenchKernel{name},
    n{n_},
    tsteps{tsteps_},
    u{new Arr2D<T>{n, n}},
    v{new Arr2D<T>{n, n}},
    p{new Arr2D<T>{n, n}},
    q{new Arr2D<T>{n, n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->u.get(), o->u.get(), eps);
  }
};
} // Base
#endif
