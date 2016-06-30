#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/gemver.hpp"
#include "PolyBench/C++/OpenMP/gemver.hpp"
#include "PolyBench/RAJA/Base/gemver.hpp"
#include "PolyBench/RAJA/OpenMP/gemver.hpp"

using DataType = Base::gemver<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::gemver<DataType>,
                           CPlusPlus::OpenMP::gemver<DataType>,
                           RAJA::Base::gemver<DataType>,
                           RAJA::OpenMP::gemver<DataType>>;

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
