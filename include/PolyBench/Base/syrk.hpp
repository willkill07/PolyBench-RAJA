#ifndef _BASE_SYRK_HPP_
#define _BASE_SYRK_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class syrk : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int m, n;
  T alpha, beta;
  std::shared_ptr<Arr2D<T>> A, C;

  syrk(std::string name, int m_, int n_)
  : PolyBenchKernel{name},
    m{m_},
    n{n_},
    alpha{static_cast<T>(1.5)},
    beta{static_cast<T>(1.2)},
    A{new Arr2D<T>{n, m}},
    C{new Arr2D<T>{n, n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->C.get(), o->C.get(), eps);
  }
};
} // Base
#endif
