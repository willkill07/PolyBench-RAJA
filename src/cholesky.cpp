#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/cholesky.hpp"
#include "PolyBench/C++/OpenMP/cholesky.hpp"
#include "PolyBench/RAJA/Base/cholesky.hpp"
#include "PolyBench/RAJA/OpenMP/cholesky.hpp"

using DataType = Base::cholesky<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::cholesky<DataType>,
                           CPlusPlus::OpenMP::cholesky<DataType>,
                           RAJA::Base::cholesky<DataType>,
                           RAJA::OpenMP::cholesky<DataType>>;

using Args = GetArgs<Kernels>;
constexpr int ARGC = CountArgs<Kernels>::value;

int main(int argc, char **argv) {
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
