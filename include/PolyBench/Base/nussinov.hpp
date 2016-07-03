#ifndef _BASE_NUSSINOV_HPP_
#define _BASE_NUSSINOV_HPP_

#include "PolyBenchKernel.hpp"

using base_t = char;

namespace Base
{
class nussinov : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = void;

  int n;
  std::shared_ptr<Arr2D<int>> table;
  std::shared_ptr<Arr1D<base_t>> seq;

  nussinov(std::string name, int n_)
  : PolyBenchKernel{name},
    n{n_},
    table{new Arr2D<int>{n, n}},
    seq{new Arr1D<base_t>{n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = int{0};
    return o && Arr2D<int>::compare(this->table.get(), o->table.get(), eps);
  }
};
} // Base
#endif
