#ifndef _ALIGNED_MEMORY_HPP_
#define _ALIGNED_MEMORY_HPP_

#include <cinttypes>

namespace mem {

  struct AlignedAllocator {
    void* operator()(::std::size_t size, ::std::size_t alignment);
  };

  struct AlignedDeleter {
    void operator()(void* p);
  };

  extern AlignedAllocator defaultAllocator;
  extern AlignedDeleter defaultDeleter;
}

#endif
