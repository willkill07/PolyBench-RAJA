#ifndef _RAJA_BASE_COVARIANCE_HPP_
#define _RAJA_BASE_COVARIANCE_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/covariance.hpp"

namespace RAJA
{
namespace Base
{
template <typename T>
class covariance : public ::Base::covariance<T>
{
  using Parent = ::Base::covariance<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  covariance(Args... args)
  : ::Base::covariance<T>{std::string{"COVARIANCE - RAJA Base"}, args...}
  {
  }

  virtual void init()
  {
    USE(READ, m, n);
    USE(READWRITE, data);
    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<init_pol>(
      RangeSegment{0, n},
      RangeSegment{0, m},
      [=](int i, int j) { data->at(i, j) = (static_cast<T>(i) * j) / m; });
  }

  virtual void exec()
  {
    USE(READ, m, n, float_n);
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    {
      USE(READ, data);
      USE(READWRITE, mean);
      forall<simd_exec>(0, m, [=](int j) {
        mean->at(j) = T(0.0);
        forall<simd_exec>(0, n, [=](int i) { mean->at(j) += data->at(i, j); });
        mean->at(j) /= float_n;
      });
    }
    {
      USE(READ, mean);
      USE(READWRITE, data);
      forallN<exec_pol>(
        RangeSegment{0, n},
        RangeSegment{0, m},
        [=](int i, int j) { data->at(i, j) -= mean->at(j); });
    }
    {
      USE(READ, data);
      USE(READWRITE, cov);
      forallN<exec_pol>(
        RangeSegment{0, m},
        RangeSegment{0, m},
        [=](int i, int j) {
          if (j >= i) {
            cov->at(i, j) = static_cast<T>(0.0);
            forall<simd_exec>(0, n, [=](int k) {
              cov->at(i, j) += data->at(k, i) * data->at(k, j);
            });
            cov->at(i, j) = cov->at(j, i) = cov->at(i, j) / (float_n - T(1.0));
          }
        });
    }
  }
};
} // Base
} // RAJA
#endif
