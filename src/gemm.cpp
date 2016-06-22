#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/gemm.hpp"
#include "C++/gemm.hpp"
#include "RAJA/gemm.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: \n  ./program <ni> <nj> <nk>" << std::endl;
    exit(-1);
  }
  int ni = std::stoi(argv[1]);
  int nj = std::stoi(argv[2]);
  int nk = std::stoi(argv[3]);

  PolyBenchKernel *vanilla = new CPlusPlus::gemm<double>{ni, nj, nk};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::gemm<double>{ni, nj, nk};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
