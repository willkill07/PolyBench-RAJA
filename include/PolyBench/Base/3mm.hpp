#ifndef _BASE_3MM_HPP_
#define _BASE_3MM_HPP_

#include "PolyBenchKernel.hpp"

namespace Base {
template <typename T>
class mm3 : public PolyBenchKernel {
public:
  using args = std::tuple<int, int, int, int, int>;
  using arg_count = std::tuple_size<args>::type;
  using default_datatype = double;

  Arr2D<T> *A, *B, *C, *D, *E, *F, *G;
  int ni, nj, nk, nl, nm;

  mm3(std::string n, int ni_, int nj_, int nk_, int nl_, int nm_)
      : PolyBenchKernel{n}, ni{ni_}, nj{nj_}, nk{nk_}, nl{nl_}, nm{nm_} {
    E = new Arr2D<T>{ni, nj};
    A = new Arr2D<T>{ni, nk};
    B = new Arr2D<T>{nk, nj};
    F = new Arr2D<T>{nj, nl};
    C = new Arr2D<T>{nj, nm};
    D = new Arr2D<T>{nm, nl};
    G = new Arr2D<T>{ni, nl};
  }

  ~mm3() {
    delete A;
    delete B;
    delete C;
    delete D;
    delete E;
    delete F;
    delete G;
  }

  virtual bool compare(const PolyBenchKernel *other) {
    return Arr2D<T>::compare(
      this->G, dynamic_cast<const mm3 *>(other)->G, static_cast<T>(1.0e-3));
  }
};
} // Base
#endif
