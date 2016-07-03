#ifndef _BASE_2MM_HPP_
#define _BASE_2MM_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class mm2 : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int, int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int ni, nj, nk, nl;
  T alpha, beta;
  std::shared_ptr<Arr2D<T>> A, B, C, D, tmp;

  mm2(std::string n, int ni_, int nj_, int nk_, int nl_)
  : PolyBenchKernel{n},
    ni{ni_},
    nj{nj_},
    nk{nk_},
    nl{nl_},
    alpha{static_cast<T>(1.5)},
    beta{static_cast<T>(1.2)},
    A{new Arr2D<T>{ni, nk}},
    B{new Arr2D<T>{nk, nj}},
    C{new Arr2D<T>{nj, nl}},
    D{new Arr2D<T>{ni, nl}},
    tmp{new Arr2D<T>{ni, nj}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->D.get(), o->D.get(), eps);
  }
};
} // Base
#endif
