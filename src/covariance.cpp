#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/covariance.hpp"
#include "PolyBench/C++/OpenMP/covariance.hpp"
#include "PolyBench/RAJA/Base/covariance.hpp"
#include "PolyBench/RAJA/OpenMP/covariance.hpp"

using DataType = Base::covariance<dummy_t>::default_datatype;
using Kernels = std::tuple<CPlusPlus::Base::covariance<DataType>,
                           CPlusPlus::OpenMP::covariance<DataType>,
                           RAJA::Base::covariance<DataType>,
                           RAJA::OpenMP::covariance<DataType>>;

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
