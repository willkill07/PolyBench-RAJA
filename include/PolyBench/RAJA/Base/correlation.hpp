#ifndef _RAJA_BASE_CORRELATION_HPP_
#define _RAJA_BASE_CORRELATION_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/correlation.hpp"

namespace RAJA
{
namespace Base
{
template <typename T>
class correlation : public ::Base::correlation<T>
{
  using Parent = ::Base::correlation<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  correlation(Args... args)
  : ::Base::correlation<T>{std::string{"CORRELATION - RAJA Base"}, args...}
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
      [=](int i, int j) { data->at(i, j) = static_cast<T>(i * j) / m + i; });
  }

  virtual void exec()
  {
    USE(READ, m, n, float_n);
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;

    T eps = static_cast<T>(0.1);
    {
      USE(READ, data);
      USE(READWRITE, mean);
      forall<simd_exec>(0, m, [=](int j) {
        mean->at(j) = static_cast<T>(0.0);
        forall<simd_exec>(0, n, [=](int i) { mean->at(j) += data->at(i, j); });
        mean->at(j) /= float_n;
      });
    }
    {
      USE(READ, data, mean);
      USE(READWRITE, stddev);
      forall<simd_exec>(0, m, [=](int j) {
        stddev->at(j) = static_cast<T>(0.0);
        forall<simd_exec>(0, n, [=](int i) {
          stddev->at(j) +=
            (data->at(i, j) - mean->at(j)) * (data->at(i, j) - mean->at(j));
        });
        stddev->at(j) /= float_n;
        stddev->at(j) = sqrt(stddev->at(j));
        stddev->at(j) =
          (stddev->at(j) <= eps) ? static_cast<T>(1.0) : stddev->at(j);
      });
    }
    {
      USE(READ, mean, stddev);
      USE(READWRITE, data);
      forallN<exec_pol>(
        RangeSegment{0, n},
        RangeSegment{0, m},
        [=](int i, int j) {
          data->at(i, j) =
            (data->at(i, j) - mean->at(j)) / (sqrt(float_n) * stddev->at(j));
        });
    }
    {
      USE(READ, data);
      USE(READWRITE, corr);
      forallN<exec_pol>(
        RangeSegment{0, m - 1},
        RangeSegment{0, m},
        [=](int i, int j) {
          if (i == j) {
            corr->at(i, i) = static_cast<T>(1.0);
          } else if (j > i) {
            corr->at(i, j) = static_cast<T>(0.0);
            forall<simd_exec>(0, n, [=](int k) {
              corr->at(i, j) += data->at(k, i) * data->at(k, j);
            });
            corr->at(j, i) = corr->at(i, j);
          }
        });
      corr->at(m - 1, m - 1) = static_cast<T>(1.0);
    }
  }
};
} // Base
} // RAJA
#endif
