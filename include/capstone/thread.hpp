#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
typedef DWORD thread_ret_t;
typedef HANDLE pthread_t;

int pthread_create(pthread_t* out, void* unused, thread_ret_t (*func)(void*), void* arg);
int pthread_join(pthread_t thread, void* unused);

#else
#include <pthread.h>
typedef void* thread_ret_t;
#endif