#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/atax.hpp"
#include "PolyBench/C++/OpenMP/atax.hpp"
#include "PolyBench/RAJA/Base/atax.hpp"
#include "PolyBench/RAJA/OpenMP/atax.hpp"

using DataType = Base::atax<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::atax<DataType>,
                           CPlusPlus::OpenMP::atax<DataType>,
                           RAJA::Base::atax<DataType>,
                           RAJA::OpenMP::atax<DataType>>;

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
