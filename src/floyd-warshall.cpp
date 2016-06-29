#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/floyd-warshall.hpp"
#include "PolyBench/C++/OpenMP/floyd-warshall.hpp"
#include "PolyBench/RAJA/Base/floyd-warshall.hpp"
#include "PolyBench/RAJA/OpenMP/floyd-warshall.hpp"

using DataType = Base::floyd_warshall<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::floyd_warshall<DataType>,
                           CPlusPlus::OpenMP::floyd_warshall<DataType>,
                           RAJA::Base::floyd_warshall<DataType>,
                           RAJA::OpenMP::floyd_warshall<DataType>>;

using Args = GetArgs<Kernels>;
constexpr int ARGC = CountArgs<Kernels>::value;

int main(int argc, char **argv) {
  if (argc != ARGC + 1) {
    std::cerr << "Invalid number of parameters (expected " << ARGC << ")"
              << std::endl;
    return EXIT_FAILURE;
  }
  auto args = parseArgs<Args>(argv + 1);
  KernelPacker versions;
  versions.addAll<Kernels>(args);
  versions.run();
  return versions.check();
}
