#ifndef _BASE_LU_HPP_
#define _BASE_LU_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class lu : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  std::shared_ptr<Arr2D<T>> A;

  lu(std::string name, int n_)
  : PolyBenchKernel{name}, n{n_}, A{new Arr2D<T>{n, n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->A.get(), o->A.get(), eps);
  }
};
} // Base
#endif
