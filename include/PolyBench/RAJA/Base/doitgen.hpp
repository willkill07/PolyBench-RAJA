#ifndef _RAJA_BASE_DOITGEN_HPP_
#define _RAJA_BASE_DOITGEN_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/doitgen.hpp"

namespace RAJA {
namespace Base {
template <typename T>
class doitgen : public ::Base::doitgen<T> {
  using Parent = ::Base::doitgen<T>;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  doitgen(Args... args) : ::Base::doitgen<T>{"DOITGEN - RAJA Base", args...} {
  }

  virtual void init() {
    USE(READ, nr, nq, np);
    USE(READWRITE, A, C4);
    using init_pol = NestedPolicy<ExecList<simd_exec, simd_exec, simd_exec>>;
    forallN<init_pol>(
      RangeSegment{0, nr},
      RangeSegment{0, nq},
      RangeSegment{0, np},
      [=](int i, int j, int k) {
        A->at(i, j, k) = static_cast<T>((i * j + k) % np) / np;
      });
    using init_pol2 = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<init_pol2>(
      RangeSegment{0, np},
      RangeSegment{0, np},
      [=](int i, int j) { C4->at(i, j) = static_cast<T>(i * j % np) / np; });
  }

  virtual void exec() {
    USE(READ, nr, nq, np, C4);
    USE(READWRITE, sum, A);
    using exec_pol = NestedPolicy<ExecList<simd_exec, simd_exec>>;
    forallN<exec_pol>(
      RangeSegment{0, nr},
      RangeSegment{0, nq},
      [=](int r, int q) {
        forall<simd_exec>(0, np, [=](int p) {
          sum->at(r, q, p) = static_cast<T>(0.0);
          forall<simd_exec>(0, np, [=](int s) {
            sum->at(r, q, p) += A->at(r, q, s) * C4->at(s, p);
          });
        });
        forall<simd_exec>(0, np, [=](int p) {
          A->at(r, q, p) = sum->at(r, q, p);
        });
      });
  }
};
} // Base
} // RAJA
#endif
