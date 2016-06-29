#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/correlation.hpp"
#include "PolyBench/C++/OpenMP/correlation.hpp"
#include "PolyBench/RAJA/Base/correlation.hpp"
#include "PolyBench/RAJA/OpenMP/correlation.hpp"

using DataType = Base::correlation<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::correlation<DataType>,
                           CPlusPlus::OpenMP::correlation<DataType>,
                           RAJA::Base::correlation<DataType>,
                           RAJA::OpenMP::correlation<DataType>>;

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
