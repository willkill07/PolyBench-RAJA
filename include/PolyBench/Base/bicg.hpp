#ifndef _BASE_BICG_HPP_
#define _BASE_BICG_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class bicg : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int m, n;
  std::shared_ptr<Arr2D<T>> A;
  std::shared_ptr<Arr1D<T>> s, p, q, r;

  bicg(std::string name, int m_, int n_)
  : PolyBenchKernel{name},
    m{m_},
    n{n_},
    A{new Arr2D<T>{n, m}},
    s{new Arr1D<T>{m}},
    q{new Arr1D<T>{n}},
    p{new Arr1D<T>{m}},
    r{new Arr1D<T>{n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr1D<T>::compare(this->s.get(), o->s.get(), eps)
           && Arr1D<T>::compare(this->q.get(), o->q.get(), eps);
  }
};
} // Base
#endif
