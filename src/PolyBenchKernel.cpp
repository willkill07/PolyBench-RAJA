#include "PolyBenchKernel.hpp"
#include "Timer.hpp"

PolyBenchKernel::PolyBenchKernel(std::string n) : name{n}
{
}
void PolyBenchKernel::run()
{
  std::cout << "[" << this->name << "] Initializing" << std::endl;
  this->init();
  std::cout << "[" << this->name << "] Running" << std::endl;
  {
    util::BlockTimer t{this->name};
    this->exec();
  }
  this->teardown();
  std::cout << "[" << this->name << "] Done" << std::endl;
}
