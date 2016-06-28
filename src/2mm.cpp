#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/2mm.hpp"
#include "C++/2mm.hpp"
#include "RAJA/2mm.hpp"

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: \n  ./program <ni> <nj> <nk> <nl>" << std::endl;
    exit(-1);
  }
  int ni = std::stoi(argv[1]);
  int nj = std::stoi(argv[2]);
  int nk = std::stoi(argv[3]);
  int nl = std::stoi(argv[4]);

  PolyBenchKernel *vanilla = new CPlusPlus::mm2<double>{ni, nj, nk, nl};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::mm2<double>{ni, nj, nk, nl};
  raja->run();

  bool diff = !vanilla->compare(raja);
  if (diff)
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return (diff);
}
