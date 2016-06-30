#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/syrk.hpp"
#include "PolyBench/C++/OpenMP/syrk.hpp"
#include "PolyBench/RAJA/Base/syrk.hpp"
#include "PolyBench/RAJA/OpenMP/syrk.hpp"

using DataType = Base::syrk<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::syrk<DataType>,
                           CPlusPlus::OpenMP::syrk<DataType>,
                           RAJA::Base::syrk<DataType>,
                           RAJA::OpenMP::syrk<DataType>>;

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
