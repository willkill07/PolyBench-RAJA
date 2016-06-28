#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/covariance.hpp"
#include "C++/covariance.hpp"
#include "RAJA/covariance.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: \n  ./program <n> <m>" << std::endl;
    exit(-1);
  }
  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);

  PolyBenchKernel *vanilla = new CPlusPlus::covariance<double>{n, m};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::covariance<double>{n, m};
  raja->run();

  bool diff = !vanilla->compare(raja);
  if (diff)
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return (diff);
}
