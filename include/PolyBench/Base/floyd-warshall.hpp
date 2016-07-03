#ifndef _BASE_FLOYD_WARSHALL_HPP_
#define _BASE_FLOYD_WARSHALL_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class floyd_warshall : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = int;

  int n;
  std::shared_ptr<Arr2D<T>> path;

  floyd_warshall(std::string name, int n_)
  : PolyBenchKernel{name}, n{n_}, path{new Arr2D<T>{n, n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{0};
    return o && Arr2D<T>::compare(this->path.get(), o->path.get(), eps);
  }
};
} // Base
#endif
