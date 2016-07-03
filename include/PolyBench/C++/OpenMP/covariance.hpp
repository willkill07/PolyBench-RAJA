#ifndef _CPP_OMP_COVARIANCE_HPP_
#define _CPP_OMP_COVARIANCE_HPP_

#include "PolyBench/Base/covariance.hpp"

namespace CPlusPlus
{
namespace OpenMP
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
  : ::Base::covariance<T>{std::string{"COVARIANCE - C++ OpenMP"}, args...}
  {
  }

  virtual void init()
  {
    USE(READ, m, n);
    USE(READWRITE, data);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++)
        data->at(i, j) = (static_cast<T>(i) * j) / m;
  }

  virtual void exec()
  {
    USE(READ, m, n, float_n);
    {
      USE(READ, data);
      USE(READWRITE, mean);
      for (int j = 0; j < m; j++) {
        mean->at(j) = T(0.0);
        for (int i = 0; i < n; i++)
          mean->at(j) += data->at(i, j);
        mean->at(j) /= float_n;
      }
    }
    {
      USE(READ, mean);
      USE(READWRITE, data);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
          data->at(i, j) -= mean->at(j);
    }
    {
      USE(READ, data);
      USE(READWRITE, cov);
      for (int i = 0; i < m; i++)
        for (int j = i; j < m; j++) {
          cov->at(i, j) = T(0.0);
          for (int k = 0; k < n; k++)
            cov->at(i, j) += data->at(k, i) * data->at(k, j);
          cov->at(i, j) /= (float_n - T(1.0));
          cov->at(j, i) = cov->at(i, j);
        }
    }
  }
};
} // OpenMP
} // CPlusPlus
#endif
