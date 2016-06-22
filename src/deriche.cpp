#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/deriche.hpp"
#include "C++/deriche.hpp"
#include "RAJA/deriche.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: \n  ./program <w> <h>" << std::endl;
    exit(-1);
  }
  int w = std::stoi(argv[1]);
  int h = std::stoi(argv[1]);

  PolyBenchKernel *vanilla = new CPlusPlus::deriche<float>{w, h};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::deriche<float>{w, h};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
