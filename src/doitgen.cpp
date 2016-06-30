#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/doitgen.hpp"
#include "PolyBench/C++/OpenMP/doitgen.hpp"
#include "PolyBench/RAJA/Base/doitgen.hpp"
#include "PolyBench/RAJA/OpenMP/doitgen.hpp"

using DataType = Base::doitgen<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::doitgen<DataType>,
                           CPlusPlus::OpenMP::doitgen<DataType>,
                           RAJA::Base::doitgen<DataType>,
                           RAJA::OpenMP::doitgen<DataType>>;

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
