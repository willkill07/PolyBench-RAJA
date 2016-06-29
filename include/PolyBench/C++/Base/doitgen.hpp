#ifndef _CPP_BASE_DOITGEN_HPP_
#define _CPP_BASE_DOITGEN_HPP_

#include "PolyBench/Base/doitgen.hpp"

namespace CPlusPlus {
namespace Base {
template <typename T>
class doitgen : public ::Base::doitgen<T> {
  using Parent = ::Base::doitgen<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  doitgen(Args... args) : ::Base::doitgen<T>{"DOITGEN - C++ Base", args...} {
  }

  virtual void init() {
    USE(READ, nr, nq, np);
    USE(READWRITE, A, C4);
    for (int i = 0; i < nr; i++)
      for (int j = 0; j < nq; j++)
        for (int k = 0; k < np; k++)
          A->at(i, j, k) = static_cast<T>((i * j + k) % np) / np;
    for (int i = 0; i < np; i++)
      for (int j = 0; j < np; j++)
        C4->at(i, j) = static_cast<T>(i * j % np) / np;
  }

  virtual void exec() {
    USE(READ, nr, nq, np, C4);
    USE(READWRITE, sum, A);
    for (int r = 0; r < nr; r++)
      for (int q = 0; q < nq; q++) {
        for (int p = 0; p < np; p++) {
          sum->at(r, q, p) = static_cast<T>(0.0);
          for (int s = 0; s < np; s++)
            sum->at(r, q, p) += A->at(r, q, s) * C4->at(s, p);
        }
        for (int p = 0; p < np; p++)
          A->at(r, q, p) = sum->at(r, q, p);
      }
  }
};
} // Base
} // CPlusPlus
#endif
