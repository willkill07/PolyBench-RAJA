#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/syr2k.hpp"
#include "PolyBench/C++/OpenMP/syr2k.hpp"
#include "PolyBench/RAJA/Base/syr2k.hpp"
#include "PolyBench/RAJA/OpenMP/syr2k.hpp"

using DataType = Base::syr2k<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::syr2k<DataType>,
                           CPlusPlus::OpenMP::syr2k<DataType>,
                           RAJA::Base::syr2k<DataType>,
                           RAJA::OpenMP::syr2k<DataType>>;

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
