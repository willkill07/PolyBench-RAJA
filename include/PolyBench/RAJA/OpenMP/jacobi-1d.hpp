#ifndef _RAJA_OMP_JACOBI_1D_HPP_
#define _RAJA_OMP_JACOBI_1D_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/jacobi-1d.hpp"

namespace RAJA
{
namespace OpenMP
{
template <typename T>
class jacobi_1d : public ::Base::jacobi_1d<T>
{
  using Parent = ::Base::jacobi_1d<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  jacobi_1d(Args... args)
  : ::Base::jacobi_1d<T>{"JACOBI-1D - RAJA OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A, B);
    forall<omp_parallel_for_exec>(0, n, [=](int i) {
      A->at(i) = (static_cast<T>(i) + 2) / n;
      B->at(i) = (static_cast<T>(i) + 3) / n;
    });
  }

  virtual void exec()
  {
    USE(READ, n, tsteps);
    USE(READWRITE, A, B);
    for (int t = 0; t < tsteps; t++) {
      forall<omp_parallel_for_exec>(1, n - 1, [=](int i) {
        B->at(i) =
          static_cast<T>(0.33333) * (A->at(i - 1) + A->at(i) + A->at(i + 1));
      });
      forall<omp_parallel_for_exec>(1, n - 1, [=](int i) {
        A->at(i) =
          static_cast<T>(0.33333) * (B->at(i - 1) + B->at(i) + B->at(i + 1));
      });
    }
  }
};
} // OpenMP
} // RAJA
#endif
