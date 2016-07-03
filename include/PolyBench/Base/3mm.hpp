#ifndef _BASE_3MM_HPP_
#define _BASE_3MM_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class mm3 : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int, int, int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int ni, nj, nk, nl, nm;
  std::shared_ptr<Arr2D<T>> A, B, C, D, E, F, G;

  mm3(std::string n, int ni_, int nj_, int nk_, int nl_, int nm_)
  : PolyBenchKernel{n},
    ni{ni_},
    nj{nj_},
    nk{nk_},
    nl{nl_},
    nm{nm_},
    A{new Arr2D<T>{ni, nk}},
    B{new Arr2D<T>{nk, nj}},
    C{new Arr2D<T>{nj, nm}},
    D{new Arr2D<T>{nm, nl}},
    E{new Arr2D<T>{ni, nj}},
    F{new Arr2D<T>{nj, nl}},
    G{new Arr2D<T>{ni, nl}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->G.get(), o->G.get(), eps);
  }
};
} // Base
#endif
