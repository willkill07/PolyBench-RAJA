#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/jacobi-1d.hpp"
#include "PolyBench/C++/OpenMP/jacobi-1d.hpp"
#include "PolyBench/RAJA/Base/jacobi-1d.hpp"
#include "PolyBench/RAJA/OpenMP/jacobi-1d.hpp"

using DataType = Base::jacobi_1d<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::jacobi_1d<DataType>,
                           CPlusPlus::OpenMP::jacobi_1d<DataType>,
                           RAJA::Base::jacobi_1d<DataType>,
                           RAJA::OpenMP::jacobi_1d<DataType>>;

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
