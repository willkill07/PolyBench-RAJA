#ifndef _BASE_GEMVER_HPP_
#define _BASE_GEMVER_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class gemver : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  T alpha{static_cast<T>(1.5)}, beta{static_cast<T>(1.2)};
  T fn;
  Arr2D<T> *A;
  Arr1D<T> *u1, *u2, *v1, *v2, *w, *x, *y, *z;

  gemver(std::string name, int n_)
      : PolyBenchKernel{name}, n{n_}, fn{static_cast<T>(n)}
  {
    A = new Arr2D<T>{n, n};
    u1 = new Arr1D<T>{n};
    u2 = new Arr1D<T>{n};
    v1 = new Arr1D<T>{n};
    v2 = new Arr1D<T>{n};
    w = new Arr1D<T>{n};
    x = new Arr1D<T>{n};
    y = new Arr1D<T>{n};
    z = new Arr1D<T>{n};
  }

  ~gemver()
  {
    delete A;
    delete u1;
    delete u2;
    delete v1;
    delete v2;
    delete w;
    delete x;
    delete y;
    delete z;
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr1D<T>::compare(
      this->w, dynamic_cast<const gemver *>(other)->w, static_cast<T>(0));
  }
};
} // Base
#endif
