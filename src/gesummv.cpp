#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/gesummv.hpp"
#include "C++/gesummv.hpp"
#include "RAJA/gesummv.hpp"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: \n  ./program <n>" << std::endl;
    exit(-1);
  }
  int n = std::stoi(argv[1]);

  PolyBenchKernel *vanilla = new CPlusPlus::gesummv<double>{n};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::gesummv<double>{n};
  raja->run();

  bool diff = !vanilla->compare(raja);
  if (diff)
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return (diff);
}
