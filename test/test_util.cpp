#include "gtest/gtest.h"
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

using namespace std;

/* Utils */
template <typename T>
void ASSERT_ARRAY_EQ(T* expected, T* actual, int n)
{
    bool failed = false;
    int n_different = 0;
    for (int i = 0; i < n; i++) {
        if (expected[i] != actual[i]) {
            cout << "expected[" << i << "] = " << expected[i] << ", actual[" << i << "] = " << actual[i] << endl;
            failed = true;
            n_different++;
        }
    }
    if (failed) {
        cout << "Number of different elements: " << n_different << endl;
        FAIL();
    }
}

template <typename T>
void ASSERT_ARRAY_EQ(T expected, T* actual, int n)
{
    bool failed = false;
    int n_different = 0;
    for (int i = 0; i < n; i++) {
        if (expected != actual[i]) {
            cout << "expected: " << expected << ", actual[" << i << "] = " << actual[i] << endl;
            failed = true;
            n_different++;
        }
    }
    if (failed) {
        cout << "Number of different elements: " << n_different << ", " << (float)n_different / (float)n * 100 << "%" << endl;
        FAIL();
    }
}

void shuffle_bytes(uint8_t* bytes, int len)
{
    for (int i = 0; i < len; i++) {
        int j = rand() % len;
        uint8_t tmp = bytes[i];
        bytes[i] = bytes[j];
        bytes[j] = tmp;
    }
}

#define BROADCAST(ptr, val, len)    \
    for (int i = 0; i < (len); i++) \
    (ptr)[i] = (val)

#define SETUP_DEQUANT_TENSORS(n)                               \
    float* in = (float*)_mm_malloc(n * sizeof(float), 64);     \
    int8_t* out = (int8_t*)_mm_malloc(n * sizeof(int8_t), 64); \
    float* out_s = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);

#define TEARDOWN_DEQUANT_TENSORS() \
    _mm_free(in);                  \
    _mm_free(out);                 \
    _mm_free(out_s)

#define SETUP_TENSORS(m, n)                                                 \
    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);                       \
    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64); \
    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);         \
    int8_t* in = (int8_t*)_mm_malloc(n, 64);                                \
    float* in_s = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);  \
    float* out = (float*)_mm_malloc(m * sizeof(float), 64)

#define TEARDOWN_TENSORS() \
    _mm_free(w);           \
    _mm_free(s);           \
    _mm_free(z);           \
    _mm_free(in);          \
    _mm_free(in_s);        \
    _mm_free(out)
/* Utils */