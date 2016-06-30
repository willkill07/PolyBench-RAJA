#ifndef _BASE_MVT_HPP_
#define _BASE_MVT_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class mvt : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  Arr2D<T> *A;
  Arr1D<T> *x1, *x2, *y_1, *y_2;

  mvt(std::string name, int n_) : PolyBenchKernel{name}, n{n_}
  {
    A = new Arr2D<T>{n, n};
    x1 = new Arr1D<T>{n};
    x2 = new Arr1D<T>{n};
    y_1 = new Arr1D<T>{n};
    y_2 = new Arr1D<T>{n};
  }

  ~mvt()
  {
    delete A;
    delete x1;
    delete x2;
    delete y_1;
    delete y_2;
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr1D<T>::compare(
             this->x1,
             dynamic_cast<const mvt *>(other)->x1,
             static_cast<T>(1.0e-3))
           && Arr1D<T>::compare(
                this->x2,
                dynamic_cast<const mvt *>(other)->x2,
                static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
