#ifndef _CPP_BASE_3MM_HPP_
#define _CPP_BASE_3MM_HPP_

#include "PolyBench/Base/3mm.hpp"

namespace CPlusPlus {
namespace Base {
template <typename T>
class mm3 : public ::Base::mm3<T> {
  using Parent = ::Base::mm3<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  mm3(Args... args) : ::Base::mm3<T>{std::string{"3MM - C++ Base"}, args...} {
  }

  virtual void init() {
    USE(READWRITE, A, B, C, D);
    USE(READ, ni, nj, nk, nl, nm);

    for (int i = 0; i < ni; i++)
      for (int j = 0; j < nk; j++)
        A->at(i, j) = static_cast<T>((i * j + 1) % ni) / (5 * ni);
    for (int i = 0; i < nk; i++)
      for (int j = 0; j < nj; j++)
        B->at(i, j) = static_cast<T>((i * (j + 1) + 2) % nj) / (5 * nj);
    for (int i = 0; i < nj; i++)
      for (int j = 0; j < nm; j++)
        C->at(i, j) = static_cast<T>(i * (j + 3) % nl) / (5 * nl);
    for (int i = 0; i < nm; i++)
      for (int j = 0; j < nl; j++)
        D->at(i, j) = static_cast<T>((i * (j + 2) + 2) % nk) / (5 * nk);
  }

  virtual void exec() {
    USE(READWRITE, E, F, G);
    USE(READ, ni, nj, nk, nl, nm, A, B, C, D);
    for (int i = 0; i < ni; i++)
      for (int j = 0; j < nj; j++) {
        E->at(i, j) = static_cast<T>(0.0);
        for (int k = 0; k < nk; ++k)
          E->at(i, j) += A->at(i, k) * B->at(k, j);
      }
    for (int i = 0; i < nj; i++)
      for (int j = 0; j < nl; j++) {
        F->at(i, j) = static_cast<T>(0.0);
        for (int k = 0; k < nm; ++k)
          F->at(i, j) += C->at(i, k) * D->at(k, j);
      }
    for (int i = 0; i < ni; i++)
      for (int j = 0; j < nl; j++) {
        G->at(i, j) = static_cast<T>(0.0);
        for (int k = 0; k < nj; ++k)
          G->at(i, j) += E->at(i, k) * F->at(k, j);
      }
  }
};
} // Base
} // CPlusPlus
#endif
