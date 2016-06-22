#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/bicg.hpp"
#include "C++/bicg.hpp"
#include "RAJA/bicg.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: \n  ./program <m> <n>" << std::endl;
    exit(-1);
  }
  int m = std::stoi(argv[1]);
  int n = std::stoi(argv[2]);

  PolyBenchKernel *vanilla = new CPlusPlus::bicg<double>{m, n};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::bicg<double>{m, n};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
