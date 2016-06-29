#ifndef _RAJA_BASE_2MM_HPP_
#define _RAJA_BASE_2MM_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/2mm.hpp"

namespace RAJA {
namespace Base {
template <typename T>
class mm2 : public ::Base::mm2<T> {
  using Parent = ::Base::mm2<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  mm2(Args... args) : ::Base::mm2<T>{std::string{"2MM - RAJA Base"}, args...} {
  }

  virtual void init() {
    USE(READ, ni, nj, nk, nl);
    USE(READWRITE, alpha, beta, A, B, C, D);

    alpha = static_cast<T>(1.5);
    beta = static_cast<T>(1.2);

    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<init_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nk},
      [=](int i, int k) {
        A->at(i, k) = static_cast<T>((i * k + 1) % ni) / ni;
      });

    forallN<init_pol>(
      RangeSegment{0, nk},
      RangeSegment{0, nj},
      [=](int k, int j) {
        B->at(k, j) = static_cast<T>(k * (j + 1) % nj) / nj;
      });

    forallN<init_pol>(
      RangeSegment{0, nj},
      RangeSegment{0, nl},
      [=](int j, int l) {
        C->at(j, l) = static_cast<T>((j * (l + 3) + 1) % nl) / nl;
      });

    forallN<init_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nl},
      [=](int i, int l) {
        D->at(i, l) = static_cast<T>(i * (l + 2) % nk) / nk;
      });
  }

  virtual void exec() {
    USE(READ, ni, nj, nk, nl, A, B, C, alpha, beta);
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    {
      USE(READWRITE, tmp);
      forallN<exec_pol>(
        RangeSegment{0, ni},
        RangeSegment{0, nj},
        [=](int i, int j) {
          tmp->at(i, j) = static_cast<T>(0.0);
          forall<simd_exec>(0, nk, [=](int k) {
            tmp->at(i, j) += alpha * A->at(i, k) * B->at(k, j);
          });
        });
    }
    {
      USE(READWRITE, D);
      USE(READ, tmp);
      forallN<exec_pol>(
        RangeSegment{0, ni},
        RangeSegment{0, nl},
        [=](int i, int l) {
          D->at(i, l) *= beta;
          forall<simd_exec>(0, nj, [=](int j) {
            D->at(i, l) += tmp->at(i, j) * C->at(j, l);
          });
        });
    }
  }
};
} // Base
} // RAJA
#endif
