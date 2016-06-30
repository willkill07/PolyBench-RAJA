#include <array>
#include <iostream>
#include <tuple>

#include "PolyBench.hpp"
#include "PolyBench/C++/Base/nussinov.hpp"
#include "PolyBench/C++/OpenMP/nussinov.hpp"
#include "PolyBench/RAJA/Base/nussinov.hpp"
#include "PolyBench/RAJA/OpenMP/nussinov.hpp"

using Kernels = std::tuple<CPlusPlus::Base::nussinov,
                           CPlusPlus::OpenMP::nussinov,
                           RAJA::Base::nussinov,
                           RAJA::OpenMP::nussinov>;

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
