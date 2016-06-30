#ifndef _BASE_SEIDEL_2D_HPP_
#define _BASE_SEIDEL_2D_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class seidel_2d : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n, tsteps;
  Arr2D<T> *A;

  seidel_2d(std::string name, int n_, int tsteps_)
      : PolyBenchKernel{name}, n{n_}, tsteps{tsteps_}
  {
    A = new Arr2D<T>{n, n};
  }

  ~seidel_2d()
  {
    delete A;
  }
  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr2D<T>::compare(
      this->A,
      dynamic_cast<const seidel_2d *>(other)->A,
      static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
