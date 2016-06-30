#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/fdtd-2d.hpp"
#include "PolyBench/C++/OpenMP/fdtd-2d.hpp"
#include "PolyBench/RAJA/Base/fdtd-2d.hpp"
#include "PolyBench/RAJA/OpenMP/fdtd-2d.hpp"

using DataType = Base::fdtd_2d<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::fdtd_2d<DataType>,
                           CPlusPlus::OpenMP::fdtd_2d<DataType>,
                           RAJA::Base::fdtd_2d<DataType>,
                           RAJA::OpenMP::fdtd_2d<DataType>>;

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
