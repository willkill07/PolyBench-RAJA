#ifndef _CPP_BASE_2MM_HPP_
#define _CPP_BASE_2MM_HPP_

#include "PolyBench/Base/2mm.hpp"

namespace CPlusPlus {
namespace Base {
template <typename T>
class mm2 : public ::Base::mm2<T> {
  using Parent = ::Base::mm2<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  mm2(Args... args) : ::Base::mm2<T>{std::string{"2MM - C++ Base"}, args...} {
  }

  virtual void init() {
    USE(READWRITE, alpha, beta, A, B, C, D);
    USE(READ, ni, nj, nk, nl);

    alpha = 1.5;
    beta = 1.2;

    for (int i = 0; i < ni; ++i)
      for (int k = 0; k < nk; ++k)
        A->at(i, k) = static_cast<T>((i * k + 1) % ni) / ni;

    for (int k = 0; k < nk; ++k)
      for (int j = 0; j < ni; ++j)
        B->at(k, j) = static_cast<T>(k * (j + 1) % nj) / nj;

    for (int j = 0; j < nj; ++j)
      for (int l = 0; l < nl; ++l)
        C->at(j, l) = static_cast<T>((j * (l + 3) + 1) % nl) / nl;

    for (int i = 0; i < ni; ++i)
      for (int l = 0; l < nl; ++l)
        D->at(i, l) = static_cast<T>(i * (l + 2) % nk) / nk;
  }

  virtual void exec() {
    USE(READWRITE, D, tmp);
    USE(READ, ni, nj, nk, nl, A, B, C, alpha, beta);

    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j) {
        tmp->at(i, j) = static_cast<T>(0.0);
        for (int k = 0; k < nk; ++k)
          tmp->at(i, j) += alpha * A->at(i, k) * B->at(k, j);
      }
    for (int i = 0; i < ni; ++i)
      for (int l = 0; l < nl; ++l) {
        D->at(i, l) *= beta;
        for (int j = 0; j < nj; ++j)
          D->at(i, l) += tmp->at(i, j) * C->at(j, l);
      }
  }
};
} // Base
} // CPlusPlus
#endif
