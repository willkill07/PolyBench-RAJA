#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/seidel-2d.hpp"
#include "C++/seidel-2d.hpp"
#include "RAJA/seidel-2d.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: \n  ./program <tsteps> <n>" << std::endl;
    exit(-1);
  }
  int tsteps = std::stoi(argv[1]);
  int n = std::stoi(argv[2]);

  PolyBenchKernel *vanilla = new CPlusPlus::seidel_2d<double>{n, tsteps};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::seidel_2d<double>{n, tsteps};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
