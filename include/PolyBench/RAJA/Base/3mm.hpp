#ifndef _RAJA_BASE_3MM_HPP_
#define _RAJA_BASE_3MM_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/3mm.hpp"

namespace RAJA
{
namespace Base
{
template <typename T>
class mm3 : public ::Base::mm3<T>
{
  using Parent = ::Base::mm3<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  mm3(Args... args) : ::Base::mm3<T>{std::string{"3MM - RAJA Base"}, args...}
  {
  }

  virtual void init()
  {
    USE(READ, ni, nj, nk, nl, nm);
    USE(READWRITE, A, B, C, D);

    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<init_pol>(
      RangeSegment{0, ni},
      RangeSegment{0, nk},
      [=](int i, int j) {
        A->at(i, j) = static_cast<T>((i * j + 1) % ni) / (5 * ni);
      });
    forallN<init_pol>(
      RangeSegment{0, nk},
      RangeSegment{0, nj},
      [=](int i, int j) {
        B->at(i, j) = static_cast<T>((i * (j + 1) + 2) % nj) / (5 * nj);
      });
    forallN<init_pol>(
      RangeSegment{0, nj},
      RangeSegment{0, nm},
      [=](int i, int j) {
        C->at(i, j) = static_cast<T>(i * (j + 3) % nl) / (5 * nl);
      });
    forallN<init_pol>(
      RangeSegment{0, nm},
      RangeSegment{0, nl},
      [=](int i, int j) {
        D->at(i, j) = static_cast<T>((i * (j + 2) + 2) % nk) / (5 * nk);
      });
  }

  virtual void exec()
  {
    USE(READ, ni, nj, nk, nl, nm, A, B, C, D);
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    {
      USE(READWRITE, E);
      forallN<exec_pol>(
        RangeSegment{0, ni},
        RangeSegment{0, nj},
        [=](int i, int j) {
          E->at(i, j) = static_cast<T>(0.0);
          forall<simd_exec>(0, nk, [=](int k) {
            E->at(i, j) += A->at(i, k) * B->at(k, j);
          });
        });
    }
    {
      USE(READWRITE, F);
      forallN<exec_pol>(
        RangeSegment{0, nj},
        RangeSegment{0, nl},
        [=](int i, int j) {
          F->at(i, j) = static_cast<T>(0.0);
          forall<simd_exec>(0, nm, [=](int k) {
            F->at(i, j) += C->at(i, k) * D->at(k, j);
          });
        });
    }
    {
      USE(READWRITE, G);
      USE(READ, E, F);
      forallN<exec_pol>(
        RangeSegment{0, ni},
        RangeSegment{0, nl},
        [=](int i, int j) {
          G->at(i, j) = static_cast<T>(0.0);
          forall<simd_exec>(0, nj, [=](int k) {
            G->at(i, j) += E->at(i, k) * F->at(k, j);
          });
        });
    }
  }
};
} // Base
} // RAJA
#endif
