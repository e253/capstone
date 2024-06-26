#include <capstone/capstone.hpp>
#include <capstone/cl.hpp>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace std;

// =================
// Testing Utilities
// =================
#define SVMMAP(queue, svm_ptr, size) clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_ptr, size, 0, NULL, NULL)
#define SVMUNMAP(queue, svm_ptr) clEnqueueSVMUnmap(queue, svm_ptr, 0, NULL, NULL)
#define SETUP_QUANT_TENSORS(n)                                                             \
    float* in = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(float), 64);     \
    int8_t* out = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(int8_t), 64); \
    float* out_s = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n / QBLOCK_SIZE * sizeof(float), 64);

#define MAP_QUANT_TENSORS()                 \
    SVMMAP(queue, in, n * sizeof(float));   \
    SVMMAP(queue, out, n * sizeof(int8_t)); \
    SVMMAP(queue, out_s, n / QBLOCK_SIZE * sizeof(float))

#define UNMAP_QUANT_TENSORS() \
    SVMUNMAP(queue, in);      \
    SVMUNMAP(queue, out);     \
    SVMUNMAP(queue, out_s)

#define TEARDOWN_QUANT_TENSORS() \
    clSVMFree(context, in);      \
    clSVMFree(context, out);     \
    clSVMFree(context, out_s)

#define SETUP_TENSORS(m, n)                                                                             \
    uint8_t* w = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / 2, 64);                       \
    float* s = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / QBLOCK_SIZE * sizeof(float), 64); \
    uint8_t* z = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / QBLOCK_SIZE / 2, 64);         \
    int8_t* in = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, n, 64);                                \
    float* in_s = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n / QBLOCK_SIZE * sizeof(float), 64);  \
    float* out = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * sizeof(float), 64)

#define MAP_TENSORS()                                     \
    SVMMAP(queue, w, m* n / 2);                           \
    SVMMAP(queue, s, m* n / QBLOCK_SIZE * sizeof(float)); \
    SVMMAP(queue, z, m* n / QBLOCK_SIZE / 2);             \
    SVMMAP(queue, in, n);                                 \
    SVMMAP(queue, in_s, n / QBLOCK_SIZE * sizeof(float)); \
    SVMMAP(queue, out, m * sizeof(float))

#define UNMAP_TENSORS()    \
    SVMUNMAP(queue, w);    \
    SVMUNMAP(queue, s);    \
    SVMUNMAP(queue, z);    \
    SVMUNMAP(queue, in);   \
    SVMUNMAP(queue, in_s); \
    SVMUNMAP(queue, out)

#define TEARDOWN_TENSORS()    \
    clSVMFree(context, w);    \
    clSVMFree(context, s);    \
    clSVMFree(context, z);    \
    clSVMFree(context, in);   \
    clSVMFree(context, in_s); \
    clSVMFree(context, out)

#define BROADCAST(ptr, v, len)      \
    for (int i = 0; i < (len); i++) \
    (ptr)[i] = (v)

template <typename T>
void ASSERT_ARRAY_EQ(T* expected, T* actual, int n)
{
    bool failed = false;
    int n_different = 0;
    for (int i = 0; i < n; i++) {
        if (expected[i] != actual[i]) {
            cout << "expected[" << i << "] = " << expected[i] << ", actual[" << i << "] = " << actual[i] << "; diff: " << (abs(expected[i] - actual[i])) << endl;
            failed = true;
            n_different++;
        }
    }
    if (failed) {
        cout << "Number of different elements: " << n_different << "/" << n << ", " << (float)n_different / (float)n * 100 << "%" << endl;
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
    for (int it = 0; it < 10; it++) {
        for (int i = 0; i < len; i++) {
            int j = rand() % len;
            uint8_t tmp = bytes[i];
            bytes[i] = bytes[j];
            bytes[j] = tmp;
        }
    }
}

// =====
// State
// =====
static cl_context context;
static cl_command_queue queue;

// =====
// Tests
// =====
TEST(VectorAdd, Trivial)
{
    const int n = 1024;

    float* a = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(float), 64);
    float* b = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(float), 64);
    float* c = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(float), 64);

    SVMMAP(queue, a, n * sizeof(float));
    BROADCAST(a, 1.0f, n);
    SVMUNMAP(queue, a);

    SVMMAP(queue, b, n * sizeof(float));
    BROADCAST(b, 2.0f, n);
    SVMUNMAP(queue, b);

    SVMMAP(queue, c, n * sizeof(float));
    BROADCAST(c, 0.0f, n);
    SVMUNMAP(queue, c);

    cl_event event;
    cl_int err = vector_add(a, b, c, n, queue, &event);
    CL_CHECK(err);

    clWaitForEvents(1, &event);
    clFinish(queue);

    SVMMAP(queue, c, n * sizeof(float));
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != 3.0f) {
            cout << "Error: c[" << i << "] = " << c[i] << endl;
            passed = false;
        }
    }
    SVMUNMAP(queue, c);
    if (!passed) {
        FAIL();
    }

    clSVMFree(context, a);
    clSVMFree(context, b);
    clSVMFree(context, c);
}

class QuantContrived : public testing::TestWithParam<int> { };
TEST_P(QuantContrived, Positive_Below_127)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);
    MAP_QUANT_TENSORS();

    BROADCAST(in, 2.0f, n);
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    UNMAP_QUANT_TENSORS();

    cl_int err = cl_f32_qi8f32s(in, out, out_s, n, queue, nullptr);
    CL_CHECK(err);

    MAP_QUANT_TENSORS();

    ASSERT_ARRAY_EQ((int8_t)2, out, n);
    ASSERT_ARRAY_EQ(1.0f, out_s, n / QBLOCK_SIZE);

    UNMAP_QUANT_TENSORS();

    TEARDOWN_QUANT_TENSORS();
}
TEST_P(QuantContrived, Positive_And_Negative_Below_127)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);

    MAP_QUANT_TENSORS();

    BROADCAST(in, 2.0f, n / 2);
    BROADCAST(&in[n / 2], -2.0f, n / 2);
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    UNMAP_QUANT_TENSORS();

    cl_int err = cl_f32_qi8f32s(in, out, out_s, n, queue, nullptr);
    CL_CHECK(err);

    MAP_QUANT_TENSORS();

    for (int i = 0; i < n; i++) {
        EXPECT_EQ(i < n / 2 ? 2 : -2, out[i]);
    }

    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        EXPECT_FLOAT_EQ(1.0f, out_s[i]);
    }

    UNMAP_QUANT_TENSORS();

    TEARDOWN_QUANT_TENSORS();
}
TEST_P(QuantContrived, Positive_Greater_Than_127)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);

    MAP_QUANT_TENSORS();

    for (int i = 0; i < n; i++)
        in[i] = (float)i;
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    UNMAP_QUANT_TENSORS();

    cl_int err = cl_f32_qi8f32s(in, out, out_s, n, queue, nullptr);
    CL_CHECK(err)

    vector<int8_t> expected(n, 0.0f);
    vector<float> expected_s(n / QBLOCK_SIZE, 0.0f);

    for (int i = 0; i < n; i++) {
        float expected_scale = ((i / QBLOCK_SIZE + 1) * QBLOCK_SIZE - 1) / 127.0f;
        expected[i] = static_cast<int8_t>(roundf(i / expected_scale));
        expected_s[i / QBLOCK_SIZE] = expected_scale;
    }

    MAP_QUANT_TENSORS();

    ASSERT_ARRAY_EQ(expected.data(), out, n);
    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        EXPECT_FLOAT_EQ(expected_s[i], out_s[i]);
    }

    TEARDOWN_QUANT_TENSORS();
}
TEST_P(QuantContrived, Positive_Negative_Greater_Than_127)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);

    MAP_QUANT_TENSORS();

    for (int i = 0; i < n / 2; i++)
        in[i] = (float)i;
    for (int i = n / 2; i < n; i++)
        in[i] = (float)-i;
    BROADCAST(out, 0, n);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);

    UNMAP_QUANT_TENSORS();

    cl_int err = cl_f32_qi8f32s(in, out, out_s, n, queue, nullptr);
    CL_CHECK(err)

    MAP_QUANT_TENSORS();

    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        float scale = ((i + 1) * QBLOCK_SIZE - 1) / 127.0f;
        EXPECT_FLOAT_EQ(scale, out_s[i]);
    }

    vector<int8_t> expected(n, 0.0f);
    for (int i = 0; i < n; i++) {
        float expected_scale = ((i / QBLOCK_SIZE + 1) * QBLOCK_SIZE - 1) / 127.0f;
        expected[i] = static_cast<int8_t>(round((i < n / 2 ? (float)i : (float)(-i)) / expected_scale));
    }
    ASSERT_ARRAY_EQ(expected.data(), out, n);

    TEARDOWN_QUANT_TENSORS();
}

class QuantReferenceFuzz : public testing::TestWithParam<int> { };
TEST_P(QuantReferenceFuzz, Fuzz)
{
    int n = GetParam();

    SETUP_QUANT_TENSORS(n);
    int8_t* out_ref = (int8_t*)_mm_malloc(n, 64);
    float* out_s_ref = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);

    MAP_QUANT_TENSORS();

    for (int i = 0; i < n; i++)
        in[i] = (float)(rand() % 1024 - 512);
    BROADCAST(out_s, 0.0f, n / QBLOCK_SIZE);
    BROADCAST(out_s_ref, 0.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0, n);
    BROADCAST(out_ref, 0, n);

    UNMAP_QUANT_TENSORS();

    cl_int err = cl_f32_qi8f32s(in, out, out_s, n, queue, nullptr);
    CL_CHECK(err)
    ref_f32_qi8f32s(in, out_ref, out_s_ref, n);

    MAP_QUANT_TENSORS();

    ASSERT_ARRAY_EQ(out_ref, out, n);
    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        EXPECT_FLOAT_EQ(out_s_ref[i], out_s[i]);
    }

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

    MAP_TENSORS();

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    UNMAP_TENSORS();

    cl_event ev;
    cl_int err = cl_q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, queue, &ev);
    CL_CHECK(err)
    clWaitForEvents(1, &ev);

    MAP_TENSORS();
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

    MAP_TENSORS();

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

    UNMAP_TENSORS();

    cl_event ev;
    cl_int err = cl_q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, queue, &ev);
    CL_CHECK(err)
    clWaitForEvents(1, &ev);

    MAP_TENSORS();

    ASSERT_ARRAY_EQ(6144.0f * (n / 512), out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Unique_Input_Scales)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);

    MAP_TENSORS();

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    for (int i = 0; i < n / QBLOCK_SIZE; i++)
        in_s[i] = (float)(i + 1);
    BROADCAST(out, 0.0f, m);

    UNMAP_TENSORS();

    cl_event ev;
    cl_int err = cl_q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, queue, &ev);
    CL_CHECK(err)
    clWaitForEvents(1, &ev);

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

    MAP_TENSORS();

    ASSERT_ARRAY_EQ(sum.sum, out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Unique_Weights)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);
    MAP_TENSORS();

    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
    BROADCAST(in, 2, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    UNMAP_TENSORS();

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

    cl_event ev;
    cl_int err = cl_q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, queue, &ev);
    CL_CHECK(err)
    clWaitForEvents(1, &ev);

    MAP_TENSORS();
    ASSERT_ARRAY_EQ(13312.0f * (n / 512), out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Random_Zeros)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);

    MAP_TENSORS();

    BROADCAST(w, 0x55, m * n / 2);
    BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++) {
        z[i] = rand() % 256;
    }
    BROADCAST(in, 2, n);
    BROADCAST(in_s, 1.0f, n / QBLOCK_SIZE);
    BROADCAST(out, 0.0f, m);

    UNMAP_TENSORS();

    cl_event ev;
    cl_int err = cl_q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, queue, &ev);
    CL_CHECK(err)
    clWaitForEvents(1, &ev);

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

    MAP_TENSORS();

    ASSERT_ARRAY_EQ(expected, out, m);

    TEARDOWN_TENSORS();
}

TEST_P(EGEMVContrived, Shuffled_Inputs)
{
    tuple<int, int> t = GetParam();
    int m = get<0>(t);
    int n = get<1>(t);

    SETUP_TENSORS(m, n);
    MAP_TENSORS();

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

    UNMAP_TENSORS();
    cl_event ev;
    cl_int err = cl_q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, queue, &ev);
    CL_CHECK(err)
    clWaitForEvents(1, &ev);

    MAP_TENSORS();
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
    uint8_t* w_ref = (uint8_t*)_mm_malloc(m * n / 2, 64);
    float* s_ref = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_ref = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);
    int8_t* in_ref = (int8_t*)_mm_malloc(n, 64);
    float* in_s_ref = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);
    float* out_ref = (float*)_mm_malloc(m * sizeof(float), 64);

    MAP_TENSORS();

    // ranges are large enough to produce random looking values,
    // but not so large that float error ruins equality checks.
    for (int i = 0; i < m * n / 2; i++) {
        w[i] = (uint8_t)(rand() % 256);
        w_ref[i] = w[i];
    }
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++) {
        s[i] = (float)(rand() % 48);
        s_ref[i] = s[i];
    }
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++) {
        z[i] = (uint8_t)(rand() % 256);
        z_ref[i] = z[i];
    }
    for (int i = 0; i < n; i++) {
        in[i] = (int8_t)(rand() % 24 - 12);
        in_ref[i] = in[i];
    }
    for (int i = 0; i < n / QBLOCK_SIZE; i++) {
        in_s[i] = (float)(rand() % 32);
        in_s_ref[i] = in_s[i];
    }

    BROADCAST(out, 0.0f, m);
    BROADCAST(out_ref, 0.0f, m);

    UNMAP_TENSORS();
    cl_event ev;
    cl_int err = cl_q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, queue, &ev);
    CL_CHECK(err)
    clWaitForEvents(1, &ev);
    ref_q4f32s_qi8f32s_egemv(w_ref, s_ref, z_ref, in_ref, in_s_ref, out_ref, m, n);

    MAP_TENSORS();
    bool failed = false;
    int n_different = 0;
    for (int i = 0; i < m; i++) {
        if (out[i] != out_ref[i]) {
            cout << "ref[" << i << "] = " << out_ref[i] << ", opt[" << i << "] = " << out[i] << "; diff: " << (abs(out_ref[i] - out[i])) << endl;
            failed = true;
            n_different++;
        }
    }
    if (failed) {
        cout << "Number of different elements: " << n_different << "/" << m << ", " << (float)n_different / (float)m * 100 << "%" << endl;
        FAIL();
    }

    TEARDOWN_TENSORS();
    _mm_free(w_ref);
    _mm_free(s_ref);
    _mm_free(z_ref);
    _mm_free(in_ref);
    _mm_free(in_s_ref);
    _mm_free(out_ref);
}

constexpr tuple<int, int> dims[] = {
    { 512, 512 },
    { 1024, 1024 },
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
    { 2048, 10240 },
    { 4096, 14336 },
    { 14336, 4096 },
};

INSTANTIATE_TEST_SUITE_P(, EGEMVContrived, testing::ValuesIn(dims));
INSTANTIATE_TEST_SUITE_P(, EGEMVReferenceFuzz, testing::ValuesIn(dims));

int main(int argc, char** argv)
{
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform iris_graphics;
    cl::Device iris_gpu;
    for (auto& p : platforms) {
        if (p.getInfo<CL_PLATFORM_NAME>().find("Intel(R) OpenCL HD Graphics") != string::npos) {
            iris_graphics = p;
        }
    }
    if (iris_graphics() == 0) {
        cout << "Platform 'Intel(R) OpenCL HD Graphics' not found ... exiting" << endl;
        exit(1);
    }
    vector<cl::Device> iris_devices;
    iris_graphics.getDevices(CL_DEVICE_TYPE_GPU, &iris_devices);
    if (iris_devices.size() == 0) {
        cout << "Platform 'Intel(R) OpenCL HD Graphics' found 0 devices ... exiting" << endl;
        exit(1);
    }

    cl_device_id device = iris_devices[0]();
    cl_int err;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    cl_queue_properties queue_props[] = { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    CL_CHECK(err);

    srand(2);
    cl_init(context, device);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
