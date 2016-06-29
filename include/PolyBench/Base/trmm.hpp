#ifndef _BASE_TRMM_HPP_
#define _BASE_TRMM_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class trmm : public PolyBenchKernel {
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int m, n;
  T alpha{static_cast<T>(1.5)};
  Arr2D<T> *A, *B;

  trmm(std::string name, int m_, int n_) : PolyBenchKernel{name}, m{m_}, n{n_} {
    A = new Arr2D<T>{m, m};
    B = new Arr2D<T>{m, n};
  }

  ~trmm() {
    delete A;
    delete B;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->B, dynamic_cast<const trmm *>(other)->B, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
