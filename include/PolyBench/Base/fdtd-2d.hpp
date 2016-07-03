#ifndef _BASE_FDTD_2D_HPP_
#define _BASE_FDTD_2D_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class fdtd_2d : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int nx, ny, tmax;
  std::shared_ptr<Arr2D<T>> ex, ey, hz;
  std::shared_ptr<Arr1D<T>> _fict_;

  fdtd_2d(std::string name, int nx_, int ny_, int tmax_)
  : PolyBenchKernel{name},
    nx{nx_},
    ny{ny_},
    tmax{tmax_},
    ex{new Arr2D<T>{nx, ny}},
    ey{new Arr2D<T>{nx, ny}},
    hz{new Arr2D<T>{nx, ny}},
    _fict_{new Arr1D<T>{tmax}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr2D<T>::compare(this->ex.get(), o->ex.get(), eps)
           && Arr2D<T>::compare(this->ey.get(), o->ey.get(), eps)
           && Arr2D<T>::compare(this->hz.get(), o->hz.get(), eps);
  }
};
} // Base
#endif
