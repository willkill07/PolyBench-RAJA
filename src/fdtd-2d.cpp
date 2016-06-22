#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/fdtd-2d.hpp"
#include "C++/fdtd-2d.hpp"
#include "RAJA/fdtd-2d.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: \n  ./program <tmax> <nx> <ny>" << std::endl;
    exit(-1);
  }
  int tmax = std::stoi(argv[1]);
  int nx = std::stoi(argv[2]);
  int ny = std::stoi(argv[3]);

  PolyBenchKernel *vanilla = new CPlusPlus::fdtd_2d<double>{nx, ny, tmax};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::fdtd_2d<double>{nx, ny, tmax};
  raja->run();

  if (!vanilla->compare(raja))
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return 0;
}
