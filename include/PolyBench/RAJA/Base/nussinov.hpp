#ifndef _RAJA_BASE_NUSSINOV_HPP_
#define _RAJA_BASE_NUSSINOV_HPP_

#include <RAJA/RAJA.hxx>

#include "PolyBench/Base/nussinov.hpp"

namespace RAJA {
namespace Base {
class nussinov : public ::Base::nussinov {
  using Parent = ::Base::nussinov;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  nussinov(Args... args) : ::Base::nussinov{"NUSSINOV - RAJA Base", args...} {
  }

  virtual void init() {
    USE(READ, n);
    USE(READWRITE, seq, table);
    forall<simd_exec>(0, n, [=](int i) {
      seq->at(i) = static_cast<base_t>((i + 1) % 4);
    });
    forallN<NestedPolicy<ExecList<simd_exec, simd_exec>>>(
      RangeSegment{0, n},
      RangeSegment{0, n},
      [=](int i, int j) { table->at(i, j) = 0; });
  }

  virtual void exec() {
    USE(READ, n, seq);
    USE(READWRITE, table);
    forall<seq_exec>(0, n - 1, [=](int i_) {
      int i = n - (i_ + 2);
      forall<simd_exec>(i + 1, n, [=](int j) {
        if (j - 1 >= 0)
          table->at(i, j) = std::max(table->at(i, j), table->at(i, j - 1));
        if (i + 1 < n)
          table->at(i, j) = std::max(table->at(i, j), table->at(i + 1, j));
        if (j - 1 >= 0 && i + 1 < n) {
          table->at(i, j) = std::max(
            table->at(i, j),
            table->at(i + 1, j - 1)
              + ((i < j - 1) && ((seq->at(i) + seq->at(j)) == 3)));
        }
        forall<simd_exec>(i + 1, j, [=](int k) {
          table->at(i, j) =
            std::max(table->at(i, j), table->at(i, k) + table->at(k + 1, j));
        });
      });
    });
  }
};
} // Base
} // RAJA
#endif
