#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/symm.hpp"
#include "C++/symm.hpp"
#include "RAJA/symm.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: \n  ./program <m> <n>" << std::endl;
    exit(-1);
  }
  int m = std::stoi(argv[1]);
  int n = std::stoi(argv[2]);

  PolyBenchKernel *vanilla = new CPlusPlus::symm<double>{m, n};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::symm<double>{m, n};
  raja->run();

  bool diff = !vanilla->compare(raja);
  if (diff)
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return (diff);
}
