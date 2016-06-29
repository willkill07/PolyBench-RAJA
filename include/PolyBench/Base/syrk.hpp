#ifndef _BASE_SYRK_HPP_
#define _BASE_SYRK_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class syrk : public PolyBenchKernel {
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int m, n;
  T alpha{static_cast<T>(1.5)}, beta{static_cast<T>(1.2)};
  Arr2D<T> *A, *C;

  syrk(std::string name, int m_, int n_) : PolyBenchKernel{name}, m{m_}, n{n_} {
    A = new Arr2D<T>{n, m};
    C = new Arr2D<T>{n, n};
  }

  ~syrk() {
    delete A;
    delete C;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->C, dynamic_cast<const syrk *>(other)->C, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
