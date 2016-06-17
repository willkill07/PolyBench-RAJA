/* 2mm.c: this file is part of PolyBench/C */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
/* Include polybench common header. */
#include "PolyBenchRAJA.hpp"
/* Include benchmark-specific header. */
#include "2mm.hpp"

namespace Base {
  template <typename T>
  class mm2 : public PolyBenchKernel {

  public:
    T alpha, beta;
    Arr2D<T> *A, *B, *C, *D, *tmp;
    int ni, nj, nk, nl;

    mm2 (std::string n, int ni_, int nj_, int nk_, int nl_)
      : PolyBenchKernel { n },
        ni { ni_ }, nj { nj_ }, nk { nk_ }, nl { nl_ } {
          std::cerr << ni << ' ' << nj << ' ' << nk << ' ' << nl << std::endl;
          A = new Arr2D<T> {ni, nk};
          B = new Arr2D<T> {nk, nj};
          C = new Arr2D<T> {nj, nl};
          D = new Arr2D<T> {ni, nl};
          tmp = new Arr2D<T> {ni, nj};
        }

    ~mm2() {
      delete A;
      delete B;
      delete C;
      delete D;
      delete tmp;
    }

    virtual bool compare (const PolyBenchKernel* other) {
      return Arr2D<T>::compare (
        this->D,
        dynamic_cast <const mm2*> (other)->D,
        static_cast <T> (1.0e-3));
    }
  };
}

namespace RAJA {

  template <typename T>
  class mm2 : public Base::mm2<T> {

  public:
    mm2(int ni, int nj, int nk, int nl)
      : Base::mm2<T> { "2MM - RAJA", ni, nj, nk, nl } { }

    virtual void init() {
      USE(READ, ni, nj, nk, nl);
      USE(READWRITE, alpha, beta, A, B, C, D);

      alpha = 1.5;
      beta = 1.2;

      forallN<OuterIndependent2D>(RangeSegment{0, ni},
                             RangeSegment{0, nk},
                             [=](int i, int k) {
                               A->at(i, k) = (double)((i * k + 1) % ni) / ni;
                             });

      forallN<OuterIndependent2D>(RangeSegment{0, nk},
                             RangeSegment{0, nj},
                             [=](int k, int j) {
                               B->at(k, j) = (double)(k * (j + 1) % nj) / nj;
                             });

      forallN<OuterIndependent2D>(RangeSegment{0, nj},
                             RangeSegment{0, nl},
                             [=](int j, int l) {
                               C->at(j, l) =
                                 (double)((j * (l + 3) + 1) % nl) / nl;
                             });

      forallN<OuterIndependent2D>(RangeSegment{0, ni},
                             RangeSegment{0, nl},
                             [=](int i, int l) {
                               D->at(i, l) = (double)(i * (l + 2) % nk) / nk;
                             });
    }

    virtual void exec() {
      USE(READWRITE, D, tmp);
      USE(READ, ni, nj, nk, nl, A, B, C, alpha,beta);

      using ExecPolicy =
        NestedPolicy<ExecList<omp_collapse_nowait_exec,
                              omp_collapse_nowait_exec,
                              simd_exec>,
                     OMP_Parallel<
                       Tile<TileList<tile_fixed<32>,tile_fixed<32>,tile_fixed<32>>,
                            Permute<PERM_IJK> > > >;

      forallN<ExecPolicy>(RangeSegment{0, ni},
                          RangeSegment{0, nj},
                          RangeSegment{0, nk},
                          [=](int i, int j, int k) {
                            tmp->at(i, j) +=
                              alpha * A->at(i, k) * B->at(k, j);
                          });
      forallN<Independent2D> (RangeSegment{0, ni},
                              RangeSegment{0, nl},
                              [=](int i, int l) {
                                D->at(i, l) *= beta;
                              });
      forallN<ExecPolicy>(RangeSegment{0, ni},
                          RangeSegment{0, nl},
                          RangeSegment{0, nj},
                          [=](int i, int l, int j) {
                            D->at(i, l) +=
                              tmp->at(i, j) * C->at(j, l);
                          });
    }
  };
}
namespace CPlusPlus {
  template <typename T>
  class mm2 : public Base::mm2<T> {

  public:
    mm2 (int ni, int nj, int nk, int nl)
      : Base::mm2<T> { "2MM - Vanilla", ni, nj, nk, nl } { }

    virtual void init() {
      USE(READWRITE, alpha, beta, A, B, C, D);
      USE(READ, ni, nj, nk, nl);

      alpha = 1.5;
      beta = 1.2;

      for (int i = 0; i < ni; ++i)
        for (int k = 0; k < nk; ++k)
          A->at(i, k) = (double)((i * k + 1) % ni) / ni;

      for (int k = 0; k < nk; ++k)
        for (int j = 0; j < ni; ++j)
          B->at(k, j) = (double)(k * (j + 1) % nj) / nj;

      for (int j = 0; j < nj; ++j)
        for (int l = 0; l < nl; ++l)
          C->at(j, l) = (double)((j * (l + 3) + 1) % nl) / nl;

      for (int i = 0; i < ni; ++i)
        for (int l = 0; l < nl; ++l)
          D->at(i, l) = (double)(i * (l + 2) % nk) / nk;
    }

    virtual void exec() {
      USE(READWRITE, D, tmp);
      USE(READ, ni, nj, nk, nl, A, B, C, alpha, beta);

      for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nj; ++j)
          for (int k = 0; k < nk; ++k)
            tmp->at(i, j) += alpha * A->at(i, k) * B->at(k, j);
      for (int i = 0; i < ni; ++i)
        for (int l = 0; l < nl; ++l) {
          D->at(i, l) *= beta;
          for (int j = 0; j < nj; ++j)
            D->at(i, l) += tmp->at(i, j) * C->at(j, l);
        }
    }
  };
} // CPlusPlus

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: \n  ./program <ni> <nj> <nk> <nl>" << std::endl;
    exit (-1);
  }
  int ni = std::stoi (argv[1]);
  int nj = std::stoi (argv[2]);
  int nk = std::stoi (argv[3]);
  int nl = std::stoi (argv[4]);

	PolyBenchKernel* vanilla = new CPlusPlus::mm2<double> { ni, nj, nk, nl };
	vanilla->run();
	PolyBenchKernel* raja = new RAJA::mm2<double> { ni, nj, nk, nl };
	raja->run();

	if (!vanilla->compare(raja))
		std::cerr << "error beyond epsilon detected" << std::endl;

	delete raja;
	delete vanilla;

	return 0;
}
