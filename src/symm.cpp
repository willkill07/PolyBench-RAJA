#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/symm.hpp"
#include "PolyBench/C++/OpenMP/symm.hpp"
#include "PolyBench/RAJA/Base/symm.hpp"
#include "PolyBench/RAJA/OpenMP/symm.hpp"

using DataType = Base::symm<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::symm<DataType>,
                           CPlusPlus::OpenMP::symm<DataType>,
                           RAJA::Base::symm<DataType>,
                           RAJA::OpenMP::symm<DataType>>;

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
