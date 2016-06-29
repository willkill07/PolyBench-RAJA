#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/mvt.hpp"
#include "PolyBench/C++/OpenMP/mvt.hpp"
#include "PolyBench/RAJA/Base/mvt.hpp"
#include "PolyBench/RAJA/OpenMP/mvt.hpp"

using DataType = Base::mvt<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::mvt<DataType>,
                           CPlusPlus::OpenMP::mvt<DataType>,
                           RAJA::Base::mvt<DataType>,
                           RAJA::OpenMP::mvt<DataType>>;

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
