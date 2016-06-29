#ifndef _BASE_FLOYD_WARSHALL_HPP_
#define _BASE_FLOYD_WARSHALL_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class floyd_warshall : public PolyBenchKernel {
public:
  using args = std::tuple<int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = int;

  Arr2D<T> *path;
  int n;

  floyd_warshall(std::string name, int n_) : PolyBenchKernel{name}, n{n_} {
    path = new Arr2D<T>{n, n};
  }

  ~floyd_warshall() {
    delete path;
  }
  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->path,
      dynamic_cast<const floyd_warshall *>(other)->path,
      static_cast<T>(0));
  }
};
} // Base
#endif
