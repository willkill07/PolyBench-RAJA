#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/ludcmp.hpp"
#include "C++/ludcmp.hpp"
#include "RAJA/ludcmp.hpp"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: \n  ./program <n>" << std::endl;
    exit(-1);
  }
  int n = std::stoi(argv[1]);

  PolyBenchKernel *vanilla = new CPlusPlus::ludcmp<double>{n};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::ludcmp<double>{n};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
