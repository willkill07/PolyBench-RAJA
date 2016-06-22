#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/syr2k.hpp"
#include "C++/syr2k.hpp"
#include "RAJA/syr2k.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: \n  ./program <m> <n>" << std::endl;
    exit(-1);
  }
  int m = std::stoi(argv[1]);
  int n = std::stoi(argv[2]);

  PolyBenchKernel *vanilla = new CPlusPlus::syr2k<double>{m, n};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::syr2k<double>{m, n};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
