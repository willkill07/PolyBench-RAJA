#ifndef _RAJA_DURBIN_HPP_
#define _RAJA_DURBIN_HPP_

#include <RAJA/RAJA.hxx>

#include "Base/durbin.hpp"

namespace RAJA {
template <typename T>
class durbin : public Base::durbin<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 1>::type>
  durbin(Args... args) : Base::durbin<T>{"DURBIN - RAJA", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, r);
    forall<omp_parallel_for_exec>(0, n, [=](int i) { r->at(i) = (n + 1 - i); });
  }

  virtual void exec() {
    USE(READ, n, r);
    USE(READWRITE, y);
    Arr1D<T> *z = new Arr1D<T>{n};
    T _alpha, _beta, *alpha{&_alpha}, *beta{&_beta};
    y->at(0) = -r->at(0);
    *beta = static_cast<T>(1.0);
    *alpha = -r->at(0);
    forall<seq_exec>(1, n, [=](int k) {
      *beta = (1 - *alpha * *alpha) * *beta;
      ReduceSum<omp_reduce, T> sum{0.0};
      forall<omp_parallel_for_exec>(0, k, [=](int i) {
        sum += r->at(k - i - 1) * y->at(i);
      });
      *alpha = -(r->at(k) + sum) / *beta;
      forall<omp_parallel_for_exec>(0, k, [=](int i) {
        z->at(i) = y->at(i) + *alpha * y->at(k - i - 1);
      });
      forall<omp_parallel_for_exec>(0, k, [=](int i) { y->at(i) = z->at(i); });
      y->at(k) = *alpha;
    });
    delete z;
  }
};
} // RAJA
#endif
