#include "common.hpp"
#include <cstdlib>
#ifdef _WIN32
#include <malloc.h>
#endif // _WIN32

void* _alloc(size_t size, size_t alignment)
{
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif // _WIN32
}

void _free(void* ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif // _WIN32
}