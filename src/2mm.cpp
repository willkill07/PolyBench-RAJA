#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/2mm.hpp"
#include "PolyBench/C++/OpenMP/2mm.hpp"
#include "PolyBench/RAJA/Base/2mm.hpp"
#include "PolyBench/RAJA/OpenMP/2mm.hpp"

using DataType = Base::mm2<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::mm2<DataType>,
                           CPlusPlus::OpenMP::mm2<DataType>,
                           RAJA::Base::mm2<DataType>,
                           RAJA::OpenMP::mm2<DataType>>;

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
