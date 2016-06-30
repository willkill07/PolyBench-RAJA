#ifndef _BASE_BICG_HPP_
#define _BASE_BICG_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class bicg : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  Arr2D<T> *A;
  Arr1D<T> *s, *p, *q, *r;
  int m, n;

  bicg(std::string name, int m_, int n_) : PolyBenchKernel{name}, m{m_}, n{n_}
  {
    A = new Arr2D<T>{n, m};
    s = new Arr1D<T>{m};
    q = new Arr1D<T>{n};
    p = new Arr1D<T>{m};
    r = new Arr1D<T>{n};
  }

  ~bicg()
  {
    delete A;
    delete s;
    delete q;
    delete p;
    delete r;
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr1D<T>::compare(
             this->s,
             dynamic_cast<const bicg *>(other)->s,
             static_cast<T>(1.0e-3))
           && Arr1D<T>::compare(
                this->q,
                dynamic_cast<const bicg *>(other)->q,
                static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
