#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/3mm.hpp"
#include "C++/3mm.hpp"
#include "RAJA/3mm.hpp"

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: \n  ./program <ni> <nj> <nk> <nl> "
                 "<nm>"
              << std::endl;
    exit(-1);
  }
  int ni = std::stoi(argv[1]);
  int nj = std::stoi(argv[2]);
  int nk = std::stoi(argv[3]);
  int nl = std::stoi(argv[4]);
  int nm = std::stoi(argv[5]);

  PolyBenchKernel *vanilla = new CPlusPlus::mm3<double>{ni, nj, nk, nl, nm};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::mm3<double>{ni, nj, nk, nl, nm};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
