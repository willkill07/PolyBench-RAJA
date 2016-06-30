#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/trmm.hpp"
#include "PolyBench/C++/OpenMP/trmm.hpp"
#include "PolyBench/RAJA/Base/trmm.hpp"
#include "PolyBench/RAJA/OpenMP/trmm.hpp"

using DataType = Base::trmm<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::trmm<DataType>,
                           CPlusPlus::OpenMP::trmm<DataType>,
                           RAJA::Base::trmm<DataType>,
                           RAJA::OpenMP::trmm<DataType>>;

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
