#include "capstone/capstone.hpp"
#include "gtest/gtest.h"
#include <immintrin.h>
#include <iostream>

#define BROADCAST(ptr, val, len)  \
    for (int i = 0; i < len; i++) \
    ptr[i] = val

#define SETUP_DEQUANT_TENSORS(n)                               \
    float* in = (float*)_mm_malloc(n * sizeof(float), 64);     \
    int8_t* out = (int8_t*)_mm_malloc(n * sizeof(int8_t), 64); \
    float* out_s = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);

#define TEARDOWN_DEQUANT_TENSORS() \
    _mm_free(in);                  \
    _mm_free(out);                 \
    _mm_free(out_s)

TEST(NoChange, Dequant)
{
    int n = 512;
    SETUP_DEQUANT_TENSORS(n);

    BROADCAST(in, 2.0f, n);
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    ref_f32_qi8f32s(in, out, out_s, n);

    for (int i = 0; i < n; i++) {
        EXPECT_EQ(2, out[i]);
    }

    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        EXPECT_FLOAT_EQ(1.0f, out_s[i]);
    }

    TEARDOWN_DEQUANT_TENSORS();
}

#define SETUP_TENSORS(m, n)                                                     \
    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);                           \
    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);     \
    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);             \
    int8_t* in = (int8_t*)_mm_malloc(n, 64);                                    \
    float* in_scales = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64); \
    float* out = (float*)_mm_malloc(m * sizeof(float), 64)

#define TEARDOWN_TENSORS() \
    _mm_free(w);           \
    _mm_free(s);           \
    _mm_free(z);           \
    _mm_free(in);          \
    _mm_free(in_scales);   \
    _mm_free(out)

TEST(Trivial, EGEMV)
{
    int m = 512;
    int n = 512;

    SETUP_TENSORS(m, n);

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    BROADCAST(in_scales, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    ref_q4f32s_qi8f32s_egemv(w, s, z, in, in_scales, out, m, n);

    for (int i = 0; i < m; i++) {
        EXPECT_FLOAT_EQ(8192.0f, out[i]);
    }

    TEARDOWN_TENSORS();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
