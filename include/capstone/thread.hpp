#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
typedef DWORD thread_ret_t;
typedef HANDLE pthread_t;
typedef _RTL_BARRIER pthread_barrier_t;

int pthread_create(pthread_t* out, void* unused, thread_ret_t (*func)(void*), void* arg);
int pthread_join(pthread_t thread, void* unused);

int pthread_barrier_init(pthread_barrier_t* b, void* unused, int n_threads);
int pthread_barrier_destroy(pthread_barrier_t* b);
int pthread_barrier_wait(pthread_barrier_t* b);

#else
#include <pthread.h>
typedef void* thread_ret_t;
#endif