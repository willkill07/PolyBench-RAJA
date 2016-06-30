#ifndef _RAJA_BASE_GESUMMV_HPP_
#define _RAJA_BASE_GESUMMV_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/gesummv.hpp"

namespace RAJA
{
namespace Base
{
template <typename T>
class gesummv : public ::Base::gesummv<T>
{
  using Parent = ::Base::gesummv<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gesummv(Args... args) : ::Base::gesummv<T>{"GESUMMV - RAJA Base", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, A, B, x);
    forall<simd_exec>(0, n, [=](int i) {
      x->at(i) = static_cast<T>(i % n) / n;
      forall<simd_exec>(0, n, [=](int j) {
        A->at(i, j) = static_cast<T>((i * j + 1) % n) / n;
        B->at(i, j) = static_cast<T>((i * j + 2) % n) / n;
      });
    });
  }

  virtual void exec()
  {
    USE(READ, n, alpha, beta, A, B, x);
    USE(READWRITE, y, tmp);
    forall<simd_exec>(0, n, [=](int i) {
      tmp->at(i) = static_cast<T>(0.0);
      y->at(i) = static_cast<T>(0.0);
      forall<simd_exec>(0, n, [=](int j) {
        tmp->at(i) += A->at(i, j) * x->at(j);
        y->at(i) += B->at(i, j) * x->at(j);
      });
      y->at(i) = alpha * tmp->at(i) + beta * y->at(i);
    });
  }
};
} // Base
} // RAJA
#endif
