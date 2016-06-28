#include <iostream>

#include "PolyBenchRAJA.hpp"

#include "Base/doitgen.hpp"
#include "C++/doitgen.hpp"
#include "RAJA/doitgen.hpp"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: \n  ./program <nr> <nq> <np>" << std::endl;
    exit(-1);
  }
  int nr = std::stoi(argv[1]);
  int np = std::stoi(argv[2]);
  int nq = std::stoi(argv[3]);

  PolyBenchKernel *vanilla = new CPlusPlus::doitgen<double>{nr, nq, np};
  vanilla->run();
  PolyBenchKernel *raja = new RAJA::doitgen<double>{nr, nq, np};
  raja->run();

  bool diff = !vanilla->compare(raja);
  if (diff)
    std::cerr << "error beyond epsilon detected" << std::endl;

  delete raja;
  delete vanilla;

  return (diff);
}
