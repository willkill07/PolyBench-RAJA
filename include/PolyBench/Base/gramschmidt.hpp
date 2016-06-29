#ifndef _BASE_GRAMSCHMIDT_HPP_
#define _BASE_GRAMSCHMIDT_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class gramschmidt : public PolyBenchKernel {
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int m, n;
  Arr2D<T> *A, *R, *Q;

  gramschmidt(std::string name, int m_, int n_)
      : PolyBenchKernel{name}, m{m_}, n{n_} {
    A = new Arr2D<T>{m, n};
    R = new Arr2D<T>{n, n};
    Q = new Arr2D<T>{m, n};
  }

  ~gramschmidt() {
    delete A;
    delete R;
    delete Q;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
             this->A,
             dynamic_cast<const gramschmidt *>(other)->A,
             static_cast<T>(1.0e-3))
           && Arr2D<T>::compare(
                this->R,
                dynamic_cast<const gramschmidt *>(other)->R,
                static_cast<T>(1.0e-3))
           && Arr2D<T>::compare(
                this->Q,
                dynamic_cast<const gramschmidt *>(other)->Q,
                static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
