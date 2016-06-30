#ifndef _BASE_LU_HPP_
#define _BASE_LU_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class lu : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n;
  Arr2D<T> *A;

  lu(std::string name, int n_) : PolyBenchKernel{name}, n{n_}
  {
    A = new Arr2D<T>{n, n};
  }
  ~lu()
  {
    delete A;
  }
  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr2D<T>::compare(
      this->A, dynamic_cast<const lu *>(other)->A, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
