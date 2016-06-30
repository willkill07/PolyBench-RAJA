#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/durbin.hpp"
#include "PolyBench/C++/OpenMP/durbin.hpp"
#include "PolyBench/RAJA/Base/durbin.hpp"
#include "PolyBench/RAJA/OpenMP/durbin.hpp"

using DataType = Base::durbin<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::durbin<DataType>,
                           CPlusPlus::OpenMP::durbin<DataType>,
                           RAJA::Base::durbin<DataType>,
                           RAJA::OpenMP::durbin<DataType>>;

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
