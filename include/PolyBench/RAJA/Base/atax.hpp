#ifndef _RAJA_BASE_ATAX_HPP_
#define _RAJA_BASE_ATAX_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/atax.hpp"

namespace RAJA
{
namespace Base
{
template <typename T>
class atax : public ::Base::atax<T>
{
  using Parent = ::Base::atax<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  atax(Args... args) : ::Base::atax<T>{std::string{"ATAX - RAJA Base"}, args...}
  {
  }

  virtual void init()
  {
    USE(READ, m, n);
    T fn{static_cast<T>(n)};
    USE(READWRITE, x, A);
    forall<simd_exec>(0, n, [=](int i) { x->at(i) = 1 + (i / fn); });
    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<init_pol>(
      RangeSegment{0, m},
      RangeSegment{0, n},
      [=](int i, int j) { A->at(i, j) = (T)((i + j) % n) / (5 * m); });
  }

  virtual void exec()
  {
    USE(READ, m, n, A, x);
    {
      USE(READWRITE, y, tmp);
      forall<simd_exec>(0, std::max(m, n), [=](int i) {
        if (i < n)
          y->at(i) = static_cast<T>(0.0);
        if (i < m)
          tmp->at(i) = static_cast<T>(0.0);
      });
    }
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    {
      USE(READWRITE, tmp);
      forallN<exec_pol>(
        RangeSegment{0, m},
        RangeSegment{0, n},
        [=](int i, int j) { tmp->at(i) += A->at(i, j) * x->at(j); });
    }
    {
      USE(READWRITE, y);
      USE(READ, tmp);
      forallN<exec_pol>(
        RangeSegment{0, n},
        RangeSegment{0, m},
        [=](int j, int i) { y->at(j) += A->at(i, j) * tmp->at(i); });
    }
  }
};
} // Base
} // RAJA
#endif
