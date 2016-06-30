#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/ludcmp.hpp"
#include "PolyBench/C++/OpenMP/ludcmp.hpp"
#include "PolyBench/RAJA/Base/ludcmp.hpp"
#include "PolyBench/RAJA/OpenMP/ludcmp.hpp"

using DataType = Base::ludcmp<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::ludcmp<DataType>,
                           CPlusPlus::OpenMP::ludcmp<DataType>,
                           RAJA::Base::ludcmp<DataType>,
                           RAJA::OpenMP::ludcmp<DataType>>;

using Args = GetArgs<Kernels>;
constexpr int ARGC = CountArgs<Kernels>::value;

int main(int argc, char **argv)
{
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
