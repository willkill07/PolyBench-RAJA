#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/deriche.hpp"
#include "PolyBench/C++/OpenMP/deriche.hpp"
#include "PolyBench/RAJA/Base/deriche.hpp"
#include "PolyBench/RAJA/OpenMP/deriche.hpp"

using DataType = Base::deriche<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::deriche<DataType>,
                           CPlusPlus::OpenMP::deriche<DataType>,
                           RAJA::Base::deriche<DataType>,
                           RAJA::OpenMP::deriche<DataType>>;

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
