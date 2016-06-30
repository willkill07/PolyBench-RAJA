#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/seidel-2d.hpp"
#include "PolyBench/C++/OpenMP/seidel-2d.hpp"
#include "PolyBench/RAJA/Base/seidel-2d.hpp"
#include "PolyBench/RAJA/OpenMP/seidel-2d.hpp"

using DataType = Base::seidel_2d<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::seidel_2d<DataType>,
                           CPlusPlus::OpenMP::seidel_2d<DataType>,
                           RAJA::Base::seidel_2d<DataType>,
                           RAJA::OpenMP::seidel_2d<DataType>>;

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
