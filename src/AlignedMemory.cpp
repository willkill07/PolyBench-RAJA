#include <AlignedMemory.hpp>

#include <cstdlib>
#include <cinttypes>

namespace mem {

  void* AlignedAllocator::operator()(::std::size_t size, ::std::size_t alignment) {
    using ptr_t = uint64_t*;
    ptr_t block = (ptr_t)malloc(size + --alignment + sizeof(ptr_t));
    if (!block)
      return NULL;
    ptr_t o = (ptr_t)((uint64_t)(block + sizeof(ptr_t) + alignment) & ~alignment);
    ((ptr_t*)o)[-1] = block;
    return (void*)o;
  }

  void AlignedDeleter::operator()(void* p) {
    if (!p) return;
    using ptr_t = char*;
    free((void*)(((ptr_t*)p)[-1]));
  }

  AlignedAllocator defaultAllocator;
  AlignedDeleter defaultDeleter;
}
