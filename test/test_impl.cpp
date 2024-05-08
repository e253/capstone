#include "capstone/capstone.hpp"
#include "test_util.cpp"
#include "gtest/gtest.h"
#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

using namespace std;

class QuantContrived : public testing::TestWithParam<int> { };
TEST_P(QuantContrived, Positive_Below_127)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);

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

    TEARDOWN_QUANT_TENSORS();
}

TEST_P(QuantContrived, Positive_And_Negative_Below_127)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);

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

    TEARDOWN_QUANT_TENSORS();
}

TEST_P(QuantContrived, Positive_Greater_Than_127)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);

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

    TEARDOWN_QUANT_TENSORS();
}

TEST_P(QuantContrived, Positive_Negative_Greater_Than_127)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);

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

    TEARDOWN_QUANT_TENSORS();
}

class QuantReferenceFuzz : public testing::TestWithParam<int> { };
TEST_P(QuantReferenceFuzz, Fuzz)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);
    int8_t* out_ref = (int8_t*)_mm_malloc(n, 64);
    float* out_s_ref = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);

    for (int i = 0; i < n; i++)
        in[i] = (float)(rand() - RAND_MAX / 2);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);
    BROADCAST(out_s_ref, 0.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0, n);
    BROADCAST(out_ref, 0, n);

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);
    ref_f32_qi8f32s(in, out_ref, out_s_ref, n);

    ASSERT_ARRAY_EQ(out_ref, out, n);
    ASSERT_ARRAY_EQ(out_s_ref, out_s, n / QBLOCK_SIZE);

    TEARDOWN_QUANT_TENSORS();
    _mm_free(out_ref);
    _mm_free(out_s_ref);
}
INSTANTIATE_TEST_SUITE_P(, QuantContrived, testing::Values(512, 1024, 2048, 2560, 4096, 10240, 14336));
INSTANTIATE_TEST_SUITE_P(, QuantReferenceFuzz, testing::Values(512, 1024, 2048, 2560, 4096, 10240, 14336));

class EGEMVContrived : public testing::TestWithParam<tuple<int, int>> { };
TEST_P(EGEMVContrived, Trivial)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

    // make sure inputs are not touched.
    // shouldn't vary between tests
    ASSERT_ARRAY_EQ((uint8_t)0x55, w, n);
    ASSERT_ARRAY_EQ(2.0f, s, m * n / QBLOCK_SIZE);
    ASSERT_ARRAY_EQ((uint8_t)0x11, z, m * n / QBLOCK_SIZE / 2);
    ASSERT_ARRAY_EQ((int8_t)2, in, n);
    ASSERT_ARRAY_EQ(1.0f, in_s, n / QBLOCK_SIZE);

    ASSERT_ARRAY_EQ(8192.0f * (n / 512), out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Alternated_Weight_Scales_Along_Input)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);

    BROADCAST(w, 0x55, m * n / 2);
    for (int row_block = 0; row_block < m / QBLOCK_SIZE; row_block++) {
        for (int col_block = 0; col_block < n / QBLOCK_SIZE; col_block++) {
            int block_id = row_block * n / QBLOCK_SIZE + col_block;
            BROADCAST(&s[block_id * QBLOCK_SIZE], col_block % 2 == 0 ? 1.0f : 2.0f, QBLOCK_SIZE);
        }
    }
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

    ASSERT_ARRAY_EQ(6144.0f * (n / 512), out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Unique_Input_Scales)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    for (int i = 0; i < n / QBLOCK_SIZE; i++)
        in_s[i] = (float)(i + 1);
    BROADCAST(out, 0.0f, m);

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

    vector<float> expected(n, 16.0f);
    for (int i = 0; i < n / QBLOCK_SIZE; i++)
        for (int j = i * QBLOCK_SIZE; j < (i + 1) * QBLOCK_SIZE; j++)
            expected.data()[j] *= (i + 1);
    // https://en.cppreference.com/w/cpp/algorithm/for_each
    struct Sum {
        void operator()(float a) { sum += a; }
        float sum { 0 };
    };
    Sum sum = for_each(expected.begin(), expected.end(), Sum());

    ASSERT_ARRAY_EQ(sum.sum, out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Unique_Weights)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);

    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (j % 2 == 0) {
                w[(i * n + j) / 2] &= 0x0F; // dump upper 4 bits
                w[(i * n + j) / 2] |= (((j % 16) << 4) & 0xF0); // set upper 4 bits
            } else {
                w[(i * n + j) / 2] &= 0xF0; // dump lower 4 bits
                w[(i * n + j) / 2] |= ((j % 16) & 0x0F); // set lower 4 bits
            }
        }

        shuffle_bytes(&w[i * n / 2], n / 2);
    }

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

    ASSERT_ARRAY_EQ(13312.0f * (n / 512), out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Random_Zeros)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++) {
        z[i] = rand() % 256;
    }
    BROADCAST(in, 2, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

    float* expected = (float*)_mm_malloc(m * sizeof(float), 64);
    BROADCAST(expected, 0.0f, m);
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col += QBLOCK_SIZE) {
            int block_id = row / QBLOCK_SIZE * n / QBLOCK_SIZE + col / QBLOCK_SIZE;
            int logical_offset = block_id * QBLOCK_SIZE + row % QBLOCK_SIZE;

            uint8_t zero = row % 2 == 0 ? (z[logical_offset / 2] >> 4) & 0x0F : z[logical_offset / 2] & 0x0F;

            expected[row] += (5 - zero) * 4 * 128;
        }
    }

    ASSERT_ARRAY_EQ(expected, out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Shuffled_Inputs)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    for (int i = 0; i < n / 2; i++) {
        in[i] = i % 128;
    }
    for (int i = n / 2; i < n; i++)
        in[i] = -(i % 128);
    shuffle_bytes((uint8_t*)in, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

    ASSERT_ARRAY_EQ(0.0f, out, m);

    TEARDOWN_TENSORS();
}

class EGEMVReferenceFuzz : public testing::TestWithParam<tuple<int, int>> { };
TEST_P(EGEMVReferenceFuzz, Fuzz)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);
    float* out_ref = (float*)_mm_malloc(m * sizeof(float), 64);

    for (int i = 0; i < m * n / 2; i++)
        w[i] = (uint8_t)(rand() % 256);
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++)
        s[i] = (float)(rand() % 32); // smol or else float comparison get wonky
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        z[i] = (uint8_t)(rand() % 256);
    for (int i = 0; i < n; i++) {
        in[i] = -5;
        // in[i] = (float)(i % 2);
        // in[i] = (int8_t)(rand() % 128 - 256);
    }
    for (int i = 0; i < n / QBLOCK_SIZE; i++)
        in_s[i] = (float)(rand() % 32); // smol or else float comparison get wonky

    BROADCAST(out, 0.0f, m);
    BROADCAST(out_ref, 0.0f, m);

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);
    ref_q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out_ref, m, n);

    for (int i = 0; i < m; i++)
        EXPECT_FLOAT_EQ(out_ref[i], out[i]);

    TEARDOWN_TENSORS();
    _mm_free(out_ref);
}
constexpr tuple<int, int> dims[] = {
    { 512, 512 },
    { 512, 1024 },
    { 512, 2048 },
    { 512, 2560 },
    { 512, 4096 },
    { 512, 10240 },
    { 512, 14336 },
    { 1024, 2048 },
    { 1024, 2560 },
    { 1024, 4096 },
    { 1024, 10240 },
    { 1024, 14336 },
    { 2048, 2560 },
    { 2048, 4096 },
    { 2048, 10240 }
};
INSTANTIATE_TEST_SUITE_P(, EGEMVContrived, testing::ValuesIn(dims));
INSTANTIATE_TEST_SUITE_P(, EGEMVReferenceFuzz, testing::ValuesIn(dims));

int main(int argc, char** argv)
{
    srand(1);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}