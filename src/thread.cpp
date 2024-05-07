#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// https://github.com/ggerganov/ggml/blob/8cd3975bf21657c6d1e80c7c61830977b962539e/src/ggml.c#L72
// Credit to ggerganov
// this is the right way to do it
typedef HANDLE pthread_t;
typedef DWORD thread_ret_t;

int pthread_create(pthread_t* out, void* unused, thread_ret_t (*func)(void*), void* arg)
{
    (void)unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, arg, 0, NULL);
    if (handle == NULL) {
        return 1;
    }

    *out = handle;
    return 0;
}

int pthread_join(pthread_t thread, void* unused)
{
    (void)unused;
    int ret = (int)WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

#endif // we link pthread otherwise