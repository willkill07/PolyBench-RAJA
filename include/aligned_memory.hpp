#ifndef _ALIGNED_MEMORY_HPP_
#define _ALIGNED_MEMORY_HPP_

#include <cinttypes>

extern void* aligned_malloc(std::size_t size, std::size_t alignment);
extern void aligned_free (void* p);

#endif
