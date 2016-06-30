#ifndef _BASE_NUSSINOV_HPP_
#define _BASE_NUSSINOV_HPP_

#include "PolyBenchKernel.hpp"

using base_t = char;

namespace Base
{
class nussinov : public PolyBenchKernel
{
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = void;

  int n;
  Arr2D<int> *table;
  Arr1D<base_t> *seq;

  nussinov(std::string name, int n_) : PolyBenchKernel{name}, n{n_}
  {
    table = new Arr2D<int>{n, n};
    seq = new Arr1D<base_t>{n};
  }

  ~nussinov()
  {
    delete table;
    delete seq;
  }

  virtual bool compare(const PolyBenchKernel *other)
  {
    return Arr2D<int>::
      compare(this->table, dynamic_cast<const nussinov *>(other)->table, 0);
  }
};
} // Base
#endif
