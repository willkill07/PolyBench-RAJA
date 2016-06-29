#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/3mm.hpp"
#include "PolyBench/C++/OpenMP/3mm.hpp"
#include "PolyBench/RAJA/Base/3mm.hpp"
#include "PolyBench/RAJA/OpenMP/3mm.hpp"

using DataType = Base::mm3<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::mm3<DataType>,
                           CPlusPlus::OpenMP::mm3<DataType>,
                           RAJA::Base::mm3<DataType>,
                           RAJA::OpenMP::mm3<DataType>>;

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
