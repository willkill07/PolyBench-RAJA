#ifndef _CPP_CORRELATION_HPP_
#define _CPP_CORRELATION_HPP_

#include "Base/correlation.hpp"

namespace CPlusPlus {
template <typename T>
class correlation : public Base::correlation<T> {
public:
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == 2>::type>
  correlation(Args... args)
      : Base::correlation<T>{std::string{"CORRELATION - Vanilla"}, args...} {
  }

  virtual void init() {
    USE(READ, m, n);
    USE(READWRITE, data);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++)
        data->at(i, j) = static_cast<T>(i * j) / m + i;
  }

  virtual void exec() {
    USE(READ, m, n, float_n);
    T eps = static_cast<T>(0.1);
    {
      USE(READ, data);
      USE(READWRITE, mean);
      for (int j = 0; j < m; j++) {
        mean->at(j) = static_cast<T>(0.0);
        for (int i = 0; i < n; i++)
          mean->at(j) += data->at(i, j);
        mean->at(j) /= float_n;
      }
    }
    {
      USE(READ, data, mean);
      USE(READWRITE, stddev);
      for (int j = 0; j < m; j++) {
        stddev->at(j) = static_cast<T>(0.0);
        for (int i = 0; i < n; i++)
          stddev->at(j) +=
            (data->at(i, j) - mean->at(j)) * (data->at(i, j) - mean->at(j));
        stddev->at(j) /= float_n;
        stddev->at(j) = sqrt(stddev->at(j));
        stddev->at(j) =
          (stddev->at(j) <= eps) ? static_cast<T>(1.0) : stddev->at(j);
      }
    }
    {
      USE(READ, mean, stddev);
      USE(READWRITE, data);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
          data->at(i, j) -= mean->at(j);
          data->at(i, j) /= sqrt(float_n) * stddev->at(j);
        }
    }
    {
      USE(READ, data);
      USE(READWRITE, corr);
      for (int i = 0; i < m - 1; i++) {
        corr->at(i, i) = static_cast<T>(1.0);
        for (int j = i + 1; j < m; j++) {
          corr->at(i, j) = static_cast<T>(0.0);
          for (int k = 0; k < n; k++)
            corr->at(i, j) += (data->at(k, i) * data->at(k, j));
          corr->at(j, i) = corr->at(i, j);
        }
      }
      corr->at(m - 1, m - 1) = static_cast<T>(1.0);
    }
  }
};
} // CPlusPlus
#endif
