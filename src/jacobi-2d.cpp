#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/jacobi-2d.hpp"
#include "C++/jacobi-2d.hpp"
#include "RAJA/jacobi-2d.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: \n  ./program <tsteps> <n>" << std::endl;
    exit(-1);
  }
  int tsteps = std::stoi(argv[1]);
  int n = std::stoi(argv[2]);

  PolyBenchKernel *vanilla = new CPlusPlus::jacobi_2d<double>{n, tsteps};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::jacobi_2d<double>{n, tsteps};
  raja->run();

  bool diff = !vanilla->compare(raja);
  if (diff)
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return (diff);
}
