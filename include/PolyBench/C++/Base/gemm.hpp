#ifndef _CPP_BASE_GEMM_HPP_
#define _CPP_BASE_GEMM_HPP_

#include "PolyBench/Base/gemm.hpp"

namespace CPlusPlus
{
namespace Base
{
template <typename T>
class gemm : public ::Base::gemm<T>
{
  using Parent = ::Base::gemm<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  gemm(Args... args) : ::Base::gemm<T>{"GEMM - C++ Base", args...}
  {
  }

  virtual void init()
  {
    USE(READ, ni, nj, nk);
    USE(READWRITE, C, A, B);
    for (int i = 0; i < ni; i++)
      for (int j = 0; j < nj; j++)
        C->at(i, j) = static_cast<T>((i * j + 1) % ni) / ni;
    for (int i = 0; i < ni; i++)
      for (int j = 0; j < nk; j++)
        A->at(i, j) = static_cast<T>(i * (j + 1) % nk) / nk;
    for (int i = 0; i < nk; i++)
      for (int j = 0; j < nj; j++)
        B->at(i, j) = static_cast<T>(i * (j + 2) % nj) / nj;
  }

  virtual void exec()
  {
    USE(READ, ni, nj, nk, alpha, beta, A, B);
    USE(READWRITE, C);
    for (int i = 0; i < ni; i++) {
      for (int j = 0; j < nj; j++)
        C->at(i, j) *= beta;
      for (int k = 0; k < nk; k++) {
        for (int j = 0; j < nj; j++)
          C->at(i, j) += alpha * A->at(i, k) * B->at(k, j);
      }
    }
  }
};
} // Base
} // CPlusPlus
#endif
