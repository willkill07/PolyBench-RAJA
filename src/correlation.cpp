#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/correlation.hpp"
#include "C++/correlation.hpp"
#include "RAJA/correlation.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: \n  ./program <n> <m>" << std::endl;
    exit(-1);
  }
  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);

  PolyBenchKernel *vanilla = new CPlusPlus::correlation<double>{n, m};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::correlation<double>{n, m};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
