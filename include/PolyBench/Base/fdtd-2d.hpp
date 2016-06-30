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

  Arr2D<T> *ex, *ey, *hz;
  Arr1D<T> *_fict_;
  int nx, ny, tmax;

  fdtd_2d(std::string name, int nx_, int ny_, int tmax_)
      : PolyBenchKernel{name}, nx{nx_}, ny{ny_}, tmax{tmax_}
  {
    ex = new Arr2D<T>{nx, ny};
    ey = new Arr2D<T>{nx, ny};
    hz = new Arr2D<T>{nx, ny};
    _fict_ = new Arr1D<T>{tmax};
  }

  ~fdtd_2d()
  {
    delete ex;
    delete ey;
    delete hz;
    delete _fict_;
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr2D<T>::compare(
             this->ex,
             dynamic_cast<const fdtd_2d *>(other)->ex,
             static_cast<T>(1.0e-3))
           && Arr2D<T>::compare(
                this->ey,
                dynamic_cast<const fdtd_2d *>(other)->ey,
                static_cast<T>(1.0e-3))
           && Arr2D<T>::compare(
                this->hz,
                dynamic_cast<const fdtd_2d *>(other)->hz,
                static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
