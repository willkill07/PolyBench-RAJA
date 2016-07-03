#ifndef _BASE_DOITGEN_HPP_
#define _BASE_DOITGEN_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class doitgen : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int nr, nq, np;
  std::shared_ptr<Arr3D<T>> A;
  std::shared_ptr<Arr3D<T>> sum;
  std::shared_ptr<Arr2D<T>> C4;

  doitgen(std::string name, int nr_, int nq_, int np_)
  : PolyBenchKernel{name},
    nr{nr_},
    nq{nq_},
    np{np_},
    A{new Arr3D<T>{nr, nq, np}},
    sum{new Arr3D<T>{nr, nq, np}},
    C4{new Arr2D<T>{np, np}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr3D<T>::compare(this->A.get(), o->A.get(), eps);
  }
};
} // Base
#endif
