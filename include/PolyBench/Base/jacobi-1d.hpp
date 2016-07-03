#ifndef _BASE_JACOBI_1D_HPP_
#define _BASE_JACOBI_1D_HPP_

#include "PolyBenchKernel.hpp"

namespace Base
{
template <typename T>
class jacobi_1d : public PolyBenchKernel
{
public:
  using args = std::tuple<int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  int n, tsteps;
  std::shared_ptr<Arr1D<T>> A, B;

  jacobi_1d(std::string name, int n_, int tsteps_)
  : PolyBenchKernel{name},
    n{n_},
    tsteps{tsteps_},
    A{new Arr1D<T>{n}},
    B{new Arr1D<T>{n}}
  {
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    using Type = typename std::add_pointer<typename std::add_const<
      typename std::remove_reference<decltype(*this)>::type>::type>::type;
    auto o = dynamic_cast<Type>(other);
    auto eps = T{1.0e-3};
    return o && Arr1D<T>::compare(this->A.get(), o->A.get(), eps);
  }
};
} // Base
#endif
