#ifndef _BASE_DERICHE_HPP_
#define _BASE_DERICHE_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class deriche : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = float;

  int w, h;
  T alpha;
  std::shared_ptr<Arr2D<T>> imgIn, imgOut, y1, y2;

  deriche(std::string name, int w_, int h_)
  : PolyBenchKernel{name},
    w{w_},
    h{h_},
    alpha{static_cast<T>(0.25)},
    imgIn{new Arr2D<T>{w, h}},
    imgOut{new Arr2D<T>{w, h}},
    y1{new Arr2D<T>{w, h}},
    y2{new Arr2D<T>{w, h}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->imgOut.get(), o->imgOut.get(), eps);
  }
};
} // Base
#endif
