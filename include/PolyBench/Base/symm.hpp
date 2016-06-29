#ifndef _BASE_SYMM_HPP_
#define _BASE_SYMM_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class symm : public PolyBenchKernel {
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int m, n;
  T alpha{static_cast<T>(1.5)}, beta{static_cast<T>(1.2)};
  Arr2D<T> *C, *A, *B;

  symm(std::string name, int m_, int n_) : PolyBenchKernel{name}, m{m_}, n{n_} {
    C = new Arr2D<T>{m, n};
    A = new Arr2D<T>{m, m};
    B = new Arr2D<T>{m, n};
  }

  ~symm() {
    delete C;
    delete A;
    delete B;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->C, dynamic_cast<const symm *>(other)->C, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
