#ifndef _POLYBENCH_KERNEL_HPP_
#define _POLYBENCH_KERNEL_HPP_

#include <iostream>
#include <string>

#include "PolyBenchPreprocessor.hpp"

#define READWRITE(a) auto a = this->a;
#define READ(a) const auto a = this->a;

#define USE(f, ...) EVAL(MAP1(f, __VA_ARGS__, (), 0))

class PolyBenchKernel
{
  virtual bool compare(const PolyBenchKernel *) = 0;
  virtual void init() = 0;
  virtual void exec() = 0;

public:
  std::string name;
  virtual void teardown()
  {
  }
  bool compareTo(const PolyBenchKernel *other)
  {
    std::cout << "Comparing [" << this->name << "] to [" << other->name << "]"
              << std::endl;
    return this->compare(other);
  }
  explicit PolyBenchKernel(std::string n);
  virtual ~PolyBenchKernel(){};
  void run();
};

#endif
