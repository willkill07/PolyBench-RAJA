#include <PolyBenchKernel.hpp>
#include <Timer.hpp>

PolyBenchKernel::PolyBenchKernel(std::string n) : name { n } { }

void PolyBenchKernel::run() {
  this->init();
  {
    util::BlockTimer t { this->name };
    this->exec();
  }
  this->teardown();
}
