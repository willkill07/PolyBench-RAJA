#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/lu.hpp"
#include "PolyBench/C++/OpenMP/lu.hpp"
#include "PolyBench/RAJA/Base/lu.hpp"
#include "PolyBench/RAJA/OpenMP/lu.hpp"

using DataType = Base::lu<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::lu<DataType>,
                           CPlusPlus::OpenMP::lu<DataType>,
                           RAJA::Base::lu<DataType>,
                           RAJA::OpenMP::lu<DataType>>;

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
