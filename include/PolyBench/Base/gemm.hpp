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

  T alpha{static_cast<T>(1.5)};
  T beta{static_cast<T>(1.2)};
  int ni, nj, nk;
  Arr2D<T> *C, *A, *B;

  gemm(std::string name, int ni_, int nj_, int nk_)
      : PolyBenchKernel{name}, ni{ni_}, nj{nj_}, nk{nk_}
  {
    C = new Arr2D<T>{ni, nj};
    A = new Arr2D<T>{ni, nk};
    B = new Arr2D<T>{nk, nj};
  }

  ~gemm()
  {
    delete C;
    delete A;
    delete B;
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr2D<T>::compare(
      this->C, dynamic_cast<const gemm *>(other)->C, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
