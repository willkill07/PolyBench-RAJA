#ifndef _CPP_OMP_NUSSINOV_HPP_
#define _CPP_OMP_NUSSINOV_HPP_

#include "PolyBench/Base/nussinov.hpp"

namespace CPlusPlus
{
namespace OpenMP
{
class nussinov : public ::Base::nussinov
{
  using Parent = ::Base::nussinov;

public:
  template <typename... Args,
            typename = typename std::
              enable_if<sizeof...(Args) == Parent::arg_count::value>::type>
  nussinov(Args... args) : ::Base::nussinov{"NUSSINOV - C++ OpenMP", args...}
  {
  }

  virtual void init()
  {
    USE(READ, n);
    USE(READWRITE, table, seq);
    for (int i = 0; i < n; i++) {
      seq->at(i) = static_cast<base_t>((i + 1) % 4);
    }
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        table->at(i, j) = 0;
  }

  virtual void exec()
  {
    USE(READ, n, seq);
    USE(READWRITE, table);
    for (int i = n - 1; i >= 0; i--) {
      for (int j = i + 1; j < n; j++) {
        if (j - 1 >= 0)
          table->at(i, j) =
            ((table->at(i, j) >= table->at(i, j - 1)) ? table->at(i, j)
                                                      : table->at(i, j - 1));
        if (i + 1 < n)
          table->at(i, j) =
            ((table->at(i, j) >= table->at(i + 1, j)) ? table->at(i, j)
                                                      : table->at(i + 1, j));
        if (j - 1 >= 0 && i + 1 < n) {
          if (i < j - 1)
            table->at(i, j) =
              ((table->at(i, j)
                >= table->at(i + 1, j - 1)
                     + (((seq->at(i)) + (seq->at(j))) == 3 ? 1 : 0))
                 ? table->at(i, j)
                 : table->at(i + 1, j - 1)
                     + (((seq->at(i)) + (seq->at(j))) == 3 ? 1 : 0));
          else
            table->at(i, j) =
              ((table->at(i, j) >= table->at(i + 1, j - 1))
                 ? table->at(i, j)
                 : table->at(i + 1, j - 1));
        }
        for (int k = i + 1; k < j; k++) {
          table->at(i, j) =
            ((table->at(i, j) >= table->at(i, k) + table->at(k + 1, j))
               ? table->at(i, j)
               : table->at(i, k) + table->at(k + 1, j));
        }
      }
    }
  }
};
} // OpenMP
} // CPlusPlus
#endif
