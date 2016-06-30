#ifndef _BASE_DOITGEN_HPP_
#define _BASE_DOITGEN_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class doitgen : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int nr, nq, np;
  Arr3D<T> *A;
  Arr3D<T> *sum;
  Arr2D<T> *C4;

  doitgen(std::string name, int nr_, int nq_, int np_)
      : PolyBenchKernel{name}, nr{nr_}, nq{nq_}, np{np_}
  {
    A = new Arr3D<T>{nr, nq, np};
    sum = new Arr3D<T>{nr, nq, np};
    C4 = new Arr2D<T>{np, np};
  }

  ~doitgen()
  {
    delete A;
    delete sum;
    delete C4;
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr3D<T>::compare(
      this->A, dynamic_cast<const doitgen *>(other)->A, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
