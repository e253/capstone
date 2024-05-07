#include "capstone/capstone.hpp"
#include "test_util.cpp"
#include "gtest/gtest.h"
#include <cstdlib>

TEST(Dequant, Positive_Below_127)
{
    int n = 512;

    SETUP_DEQUANT_TENSORS(n);

    BROADCAST(in, 2.0f, n);
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);

    for (int i = 0; i < n; i++) {
        EXPECT_EQ(2, out[i]);
    }

    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        EXPECT_FLOAT_EQ(1.0f, out_s[i]);
    }

    TEARDOWN_DEQUANT_TENSORS();
}

TEST(Dequant, Positive_And_Negative_Below_127)
{
    int n = 1024;

    SETUP_DEQUANT_TENSORS(n);

    BROADCAST(in, 2.0f, n / 2);
    BROADCAST(&in[n / 2], -2.0f, n / 2);
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);

    for (int i = 0; i < n; i++) {
        EXPECT_EQ(i < n / 2 ? 2 : -2, out[i]);
    }

    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        EXPECT_FLOAT_EQ(1.0f, out_s[i]);
    }

    TEARDOWN_DEQUANT_TENSORS();
}

TEST(Dequant, Positive_Greater_Than_127)
{
    int n = 1024;

    SETUP_DEQUANT_TENSORS(n);

    for (int i = 0; i < n; i++)
        in[i] = (float)i;
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);

    for (int i = 0; i < n; i++) {
        float expected_scale = ((i / QBLOCK_SIZE + 1) * QBLOCK_SIZE - 1) / 127.0f;
        int8_t expected = static_cast<int8_t>(round(i / expected_scale));
        EXPECT_EQ(expected, out[i]);
    }

    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        float expected = ((i + 1) * QBLOCK_SIZE - 1) / 127.0f;
        EXPECT_FLOAT_EQ(expected, out_s[i]);
    }

    TEARDOWN_DEQUANT_TENSORS();
}

TEST(Dequant, Positive_Negative_Greater_Than_127)
{
    int n = 1024;

    SETUP_DEQUANT_TENSORS(n);

    for (int i = 0; i < n / 2; i++)
        in[i] = (float)i;
    for (int i = n / 2; i < n; i++)
        in[i] = (float)-i;
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);

    for (int i = 0; i < n; i++) {
        float expected_scale = ((i / QBLOCK_SIZE + 1) * QBLOCK_SIZE - 1) / 127.0f;
        int8_t expected = static_cast<int8_t>(round((i < n / 2 ? (float)i : (float)(-i)) / expected_scale));
        EXPECT_EQ(expected, out[i]);
    }

    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        float expected = ((i + 1) * QBLOCK_SIZE - 1) / 127.0f;
        EXPECT_FLOAT_EQ(expected, out_s[i]);
    }

    TEARDOWN_DEQUANT_TENSORS();
}

TEST(EGEMV, Trivial)
{
    int m = 512;
    int n = 512;

    SETUP_TENSORS(m, n);

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

    ASSERT_ARRAY_EQ(8192.0f, out, m);

    TEARDOWN_TENSORS();
}

int main(int argc, char** argv)
{
    srand(1);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}