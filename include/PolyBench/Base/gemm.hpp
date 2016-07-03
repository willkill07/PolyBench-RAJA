#ifndef _BASE_GEMM_HPP_
#define _BASE_GEMM_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class gemm : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int ni, nj, nk;
  T alpha, beta;
  std::shared_ptr<Arr2D<T>> C, A, B;

  gemm(std::string name, int ni_, int nj_, int nk_)
  : PolyBenchKernel{name},
    ni{ni_},
    nj{nj_},
    nk{nk_},
    alpha{static_cast<T>(1.5)},
    beta{static_cast<T>(1.2)},
    C{new Arr2D<T>{ni, nj}},
    A{new Arr2D<T>{ni, nk}},
    B{new Arr2D<T>{nk, nj}}
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
