#ifndef _BASE_GRAMSCHMIDT_HPP_
#define _BASE_GRAMSCHMIDT_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class gramschmidt : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int m, n;
  std::shared_ptr<Arr2D<T>> A, R, Q;

  gramschmidt(std::string name, int m_, int n_)
  : PolyBenchKernel{name},
    m{m_},
    n{n_},
    A{new Arr2D<T>{m, n}},
    R{new Arr2D<T>{n, n}},
    Q{new Arr2D<T>{m, n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->A.get(), o->A.get(), eps)
           && Arr2D<T>::compare(this->R.get(), o->R.get(), eps)
           && Arr2D<T>::compare(this->Q.get(), o->Q.get(), eps);
  }
};
} // Base
#endif
