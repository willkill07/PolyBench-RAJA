#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/gesummv.hpp"
#include "PolyBench/C++/OpenMP/gesummv.hpp"
#include "PolyBench/RAJA/Base/gesummv.hpp"
#include "PolyBench/RAJA/OpenMP/gesummv.hpp"

using DataType = Base::gesummv<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::gesummv<DataType>,
                           CPlusPlus::OpenMP::gesummv<DataType>,
                           RAJA::Base::gesummv<DataType>,
                           RAJA::OpenMP::gesummv<DataType>>;

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
