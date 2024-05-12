#include "capstone/capstone.hpp"
#ifndef BENCH
#include "gtest/gtest.h"
#endif
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

#define QBLOCK_SIZE 128

#define CL_CHECK(status, op)                                            \
    if (status != CL_SUCCESS) {                                         \
        cout << "OpenCL error: " << status << " During " << op << endl; \
        exit(1);                                                        \
    }
#define SVMMAP(queue, svm_ptr, size) clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_ptr, size, 0, NULL, NULL)
#define SVMUNMAP(queue, svm_ptr) clEnqueueSVMUnmap(queue, svm_ptr, 0, NULL, NULL)

static cl_context context;
static cl_kernel vec_add_kernel;
static cl_kernel q4f32s_qi8f32s_egemv_kernel;
static cl_kernel f32_qi8f32s_kernel;
static cl_command_queue queue;

#define CL_SRC(...) #__VA_ARGS__
const string cl_src = CL_SRC(

    __kernel void vec_add(__global const float* a, __global const float* b, __global float* c, const int n) {
        int i = get_global_id(0);
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }

    inline char clamp(float x) {
        return (char)(x > 127.0f ? 127 : (x < -128.0f ? -128 : round(x)));
    }

    inline char get0(uchar w) {
        return (char)((w >> 4) & 0x0F);
    }

    inline char get1(uchar w) {
        return (char)(w & 0x0F);
    }

    __kernel void f32_qi8f32s(
        __global float* restrict in,
        __global char* restrict out,
        __global float* restrict out_s,
        int n,
        int n_el_per_thread) {
        const int gid = get_global_id(0);
        const int lid = get_local_id(0);

        float max = fabs(in[gid * n_el_per_thread]);
        for (int i = gid * n_el_per_thread; i < (gid + 1) * n_el_per_thread; i++)
            max = fmax(fabs(max), fabs(in[i]));

        max = work_group_reduce_max(max);

        __local float _scale; // poor mans work_group_broadcast
        float scale;
        if (lid == 0) {
            scale = max > 127.0f ? max / 127.0f : 1.0f;
            out_s[get_group_id(0)] = scale;
            _scale = scale;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        scale = _scale;

        for (uint i = gid * n_el_per_thread; i < (gid + 1) * n_el_per_thread; i++) {
            out[i] = clamp(round(in[i] / scale));
        }
    }

    __kernel void q4f32s_qi8f32s_egemv_kernel(
        __global uchar4* restrict w,
        __global float* restrict s,
        __global uchar* restrict z,
        __global char8* restrict in,
        __global float* restrict in_scales,
        __global float* restrict out,
        int m, int n, int n_blocks_per_thread) {
        const int QBLOCK_SIZE = 128;
        const int row_sz2_block = get_local_id(1);
        const int out_qblock = get_global_id(2);

        float acc1 = 0;
        float acc2 = 0;

        for (uint qblock = 0; qblock < n_blocks_per_thread; qblock++) {
            // qblock-acc
            float _acc1 = 0;
            float _acc2 = 0;
            const int in_qblock = get_local_id(0) * n_blocks_per_thread + qblock;
            const int QBLOCK_ID = out_qblock * (n / QBLOCK_SIZE) + in_qblock;

            // Set Zeros
            uchar _zero1 = z[QBLOCK_ID * QBLOCK_SIZE / 2 + row_sz2_block * 2 / 2];
            uchar _zero2 = _zero1;
            char zero1 = (char)((_zero1 >> 4) & 0x0F);
            char zero2 = (char)(_zero2 & 0x0F);

            for (uint i = 0; i < 128; i += 8) {
                // Load Input
                char8 input = in[(in_qblock * 128 + i) / 8];

                // logical row = out_qblock * QBLOCK_SIZE + row_sz2_block * 2
                // logical col = in_qblock * QBLOCK_SIZE + i
                // each is divided by 8 to get the index. 2 values per byte, 4 bytes per index
                uchar4 weights1 = w[((out_qblock * QBLOCK_SIZE + row_sz2_block * 2) * n + in_qblock * QBLOCK_SIZE + i) / 8];

                _acc1 = mad(convert_float(get0(weights1.s0) - zero1), convert_float(input.s0), _acc1);
                _acc1 = mad(convert_float(get1(weights1.s0) - zero1), convert_float(input.s1), _acc1);
                _acc1 = mad(convert_float(get0(weights1.s1) - zero1), convert_float(input.s2), _acc1);
                _acc1 = mad(convert_float(get1(weights1.s1) - zero1), convert_float(input.s3), _acc1);
                _acc1 = mad(convert_float(get0(weights1.s2) - zero1), convert_float(input.s4), _acc1);
                _acc1 = mad(convert_float(get1(weights1.s2) - zero1), convert_float(input.s5), _acc1);
                _acc1 = mad(convert_float(get0(weights1.s3) - zero1), convert_float(input.s6), _acc1);
                _acc1 = mad(convert_float(get1(weights1.s3) - zero1), convert_float(input.s7), _acc1);

                // logical row = out_qblock * QBLOCK_SIZE + row_sz2_block * 2
                // logical col = in_qblock * QBLOCK_SIZE + i ** + 1 **
                // each is divided by 8 to get the index. 2 values per byte, 4 bytes per index
                uchar4 weights2 = w[((out_qblock * QBLOCK_SIZE + row_sz2_block * 2 + 1) * n + in_qblock * QBLOCK_SIZE + i) / 8];

                _acc2 = mad(convert_float(get0(weights2.s0) - zero2), convert_float(input.s0), _acc2);
                _acc2 = mad(convert_float(get1(weights2.s0) - zero2), convert_float(input.s1), _acc2);
                _acc2 = mad(convert_float(get0(weights2.s1) - zero2), convert_float(input.s2), _acc2);
                _acc2 = mad(convert_float(get1(weights2.s1) - zero2), convert_float(input.s3), _acc2);
                _acc2 = mad(convert_float(get0(weights2.s2) - zero2), convert_float(input.s4), _acc2);
                _acc2 = mad(convert_float(get1(weights2.s2) - zero2), convert_float(input.s5), _acc2);
                _acc2 = mad(convert_float(get0(weights2.s3) - zero2), convert_float(input.s6), _acc2);
                _acc2 = mad(convert_float(get1(weights2.s3) - zero2), convert_float(input.s7), _acc2);
            } // block process

            acc1 += (float)_acc1 * s[QBLOCK_ID * QBLOCK_SIZE + row_sz2_block * 2] * in_scales[in_qblock];
            acc2 += (float)_acc2 * s[QBLOCK_ID * QBLOCK_SIZE + row_sz2_block * 2 + 1] * in_scales[in_qblock];
        } // qblock

        __local float acc1_local[2][64];
        __local float acc2_local[2][64];
        acc1_local[get_local_id(0)][row_sz2_block] = acc1;
        acc2_local[get_local_id(0)][row_sz2_block] = acc2;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_local_id(0) == 0) {
            acc1 = acc1_local[0][row_sz2_block] + acc1_local[1][row_sz2_block];
            acc2 = acc2_local[0][row_sz2_block] + acc2_local[1][row_sz2_block];
            out[out_qblock * 128 + row_sz2_block * 2] = acc1;
            out[out_qblock * 128 + row_sz2_block * 2 + 1] = acc2;
        }
    }

    // __kernel void q4f32s_qi8f32s_egemv_v2(
    //     __global uchar4* restrict w,
    //     __global float* restrict s,
    //     __global uchar* restrict z,
    //     __global char8* restrict in,
    //     __global float* restrict in_scales,
    //     __global float* restrict out,
    //     int m, int n, int n_blocks_per_thread) {
    //     const int QBLOCK_SIZE = 128;
    //     const int row = (get_global_id(0) << 8) + get_global_id(1);

    //     float acc = 0;
    //     for (uint blk = 0; blk < n_blocks_per_thread; blk++) {
    //         half _acc = 0;
    //         uint offset =
    //         for (uint i = get_global_id(2) * n_blocks_per_thread; i < 32; i += 8) {
    //         }
    //     }
    // }

);

// NOT THREAD SAFE!
void f32_qi8f32s(float* in, int8_t* out, float* out_s, int n, int n_threads)
{
    assert(128 <= n && n <= 32768 && "n must be between 128 and 32768");
    assert(n % 128 == 0 && "n must be a multiple of 128");

    cl_int clStatus;
    clStatus = clSetKernelArgSVMPointer(f32_qi8f32s_kernel, 0, in);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - in")
    clStatus = clSetKernelArgSVMPointer(f32_qi8f32s_kernel, 1, out);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - out")
    clStatus = clSetKernelArgSVMPointer(f32_qi8f32s_kernel, 2, out_s);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - out_s")
    clStatus = clSetKernelArg(f32_qi8f32s_kernel, 3, sizeof(int), &n);

    // 4-256 elements per thread;
    int n_el_per_thread = ceil(n / 256); // 256 threads in this dimension
    while (QBLOCK_SIZE % n_el_per_thread != 0) {
        n_el_per_thread++;
    }
    clStatus = clSetKernelArg(f32_qi8f32s_kernel, 4, sizeof(int), &n_el_per_thread);

    const size_t global_work_size = n / n_el_per_thread;
    const size_t local_work_size = QBLOCK_SIZE / n_el_per_thread;
    // local group size = (n/npt) / (QBLOCK_SIZE/npt) = n / QBLOCK_SIZE

    cl_event ev;
    clStatus = clEnqueueNDRangeKernel(
        queue, f32_qi8f32s_kernel,
        1, nullptr,
        &global_work_size, &local_work_size,
        0, nullptr, &ev);
    CL_CHECK(clStatus, "f32_qi8f32s kernel invocation")

    clStatus = clWaitForEvents(1, &ev);
    CL_CHECK(clStatus, "clWaitForEvents - f32_qi8f32s")
    clStatus = clReleaseEvent(ev);
    CL_CHECK(clStatus, "clReleaseEvent - f32_qi8f32s")

    clStatus = clFinish(queue);
    CL_CHECK(clStatus, "clFinish - f32_qi8f32s")
}

// DO NOT CALL FROM MULTIPLE THREADS!
void q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_s,
    float* out,
    int m, int n,
    int n_threads)
{
    assert(m >= 512 && n >= 512 && "m and n must be at least 128");
    assert(m <= 32768 && n <= 32768 && "m and n can be at most 16384");
    assert(m % 512 == 0 && n % 512 == 0 && "m and n must be multiples of 128");

    cl_int clStatus;
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_egemv_kernel, 0, w);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - w")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_egemv_kernel, 1, s);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - s")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_egemv_kernel, 2, z);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - z")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_egemv_kernel, 3, in);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - in")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_egemv_kernel, 4, in_s);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - in_scales")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_egemv_kernel, 5, out);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - out")
    clStatus = clSetKernelArg(q4f32s_qi8f32s_egemv_kernel, 6, sizeof(int), &m);
    CL_CHECK(clStatus, "clSetKernelArg - m")
    clStatus = clSetKernelArg(q4f32s_qi8f32s_egemv_kernel, 7, sizeof(int), &n);
    CL_CHECK(clStatus, "clSetKernelArg - n")
    int n_blocks_per_thread = n / 2 / 128;
    clStatus = clSetKernelArg(q4f32s_qi8f32s_egemv_kernel, 8, sizeof(int), &n_blocks_per_thread);
    CL_CHECK(clStatus, "clSetKernelArg - n_blocks_per_thread")

    cl_event event;
    const size_t _m = m;
    // const size_t _n = n;
    //  global work size is the number of work items in each dimension
    const size_t global_work_size[] = { 2, 64, _m / 128 };
    // local work size is the number of work items in each work group
    const size_t local_work_size[] = { 2, 64, 1 };
    // n work groups per dimension is found by dividing the global work size by the local work size
    // in each dimension
    // local_id = { 0-1, 0-63, 0-m/128 - 1 }
    // each row is split in two halfs
    // each out_block is split into 64 threads doing 2 rows each.
    clStatus = clEnqueueNDRangeKernel(
        queue, q4f32s_qi8f32s_egemv_kernel,
        3, nullptr,
        global_work_size, local_work_size,
        0, nullptr, &event);
    CL_CHECK(clStatus, "q4f32s_qi8f32s_egemv_kernel invocation")

    clStatus = clWaitForEvents(1, &event);
    CL_CHECK(clStatus, "clWaitForEvents - q4f32s_qi8f32s_egemv_kernel")
    clStatus = clReleaseEvent(event);
    CL_CHECK(clStatus, "clReleaseEvent - q4f32s_qi8f32s_egemv_kernel")

    clStatus = clFinish(queue);
    CL_CHECK(clStatus, "clFinish - q4f32s_qi8f32s_egemv_kernel")
}
#ifndef BENCH
/*
    =============
       TESTING
    =============
*/
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

    cl_int clStatus;

    clStatus = clSetKernelArgSVMPointer(vec_add_kernel, 0, a);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - a")
    clStatus = clSetKernelArgSVMPointer(vec_add_kernel, 1, b);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - b")
    clStatus = clSetKernelArgSVMPointer(vec_add_kernel, 2, c);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - c")
    clSetKernelArg(vec_add_kernel, 3, sizeof(int), &n);

    cl_event event;
    const size_t global_work_size = n;
    const size_t local_work_size = 1;
    clStatus = clEnqueueNDRangeKernel(
        queue, vec_add_kernel,
        1, nullptr,
        &global_work_size, &local_work_size,
        0, nullptr, &event);
    CL_CHECK(clStatus, "vec add kernel");

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

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);

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

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);

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

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);

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

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);

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

    int n_threads = 4;
    f32_qi8f32s(in, out, out_s, n, n_threads);
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

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

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

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

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

    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

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
    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);

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
    int n_threads = 4;
    q4f32s_qi8f32s_egemv(w, s, z, in, in_s, out, m, n, n_threads);
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

/*
    =============
       TESTING
    =============
*/
#endif

void random_init_array(uint8_t* arr, int len)
{
    for (int i = 0; i < len; i++) {
        arr[i] = rand() % 128;
    }
}

void bench_llama_ffn()
{
    // down proj 4096x14336
    // gate_proj 14336x4096
    // up_proj 14336x4096
    // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
    // we do not do the elementwise multiply
    // we do not apply SwiGLU non-linearity in the hidden_dim
    // just the matmuls. skips 14k ops out of 176M total (4096 * 14336 * 3)
    cout << "Benchmarking LLAMA FFN ..." << endl;
    cout << "down_proj @ (up_proj @ x * gate_proj @ x)" << endl;
    cout << "Hidden Dim: 14436, Dim: 4096" << endl;
    cout << endl;

    // ==== up_proj ====
    uint8_t* w_up_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / 2, 64);
    float* s_up_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_up_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / QBLOCK_SIZE / 2, 64);
    // SVMMAP(queue, w_up_proj, 14336 * 4096 / 2);
    // SVMMAP(queue, s_up_proj, 14336 * 4096 / QBLOCK_SIZE * sizeof(float));
    // SVMMAP(queue, z_up_proj, 14336 * 4096 / QBLOCK_SIZE / 2);
    // random_init_array((uint8_t*)w_up_proj, 14336 * 4096 / 2);
    // random_init_array((uint8_t*)s_up_proj, 14336 * 4096 / QBLOCK_SIZE);
    // random_init_array((uint8_t*)z_up_proj, 14336 * 4096 / QBLOCK_SIZE / 2);
    // SVMUNMAP(queue, w_up_proj);
    // SVMUNMAP(queue, s_up_proj);
    // SVMUNMAP(queue, z_up_proj);

    // // ==== gate_proj ====
    uint8_t* w_gate_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / 2, 64);
    float* s_gate_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_gate_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE / 2, 64);
    // SVMMAP(queue, w_gate_proj, 4096 * 14336 / 2);
    // SVMMAP(queue, s_gate_proj, 4096 * 14336 / QBLOCK_SIZE * sizeof(float));
    // SVMMAP(queue, z_gate_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    // random_init_array((uint8_t*)w_gate_proj, 4096 * 14336 / 2);
    // random_init_array((uint8_t*)s_gate_proj, 4096 * 14336 / QBLOCK_SIZE);
    // random_init_array((uint8_t*)z_gate_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    // SVMUNMAP(queue, w_gate_proj);
    // SVMUNMAP(queue, s_gate_proj);
    // SVMUNMAP(queue, z_gate_proj);

    // // ==== down_proj ====
    uint8_t* w_down_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / 2, 64);
    float* s_down_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_down_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE / 2, 64);
    // SVMMAP(queue, w_down_proj, 4096 * 14336 / 2);
    // SVMMAP(queue, s_down_proj, 4096 * 14336 / QBLOCK_SIZE * sizeof(float));
    // SVMMAP(queue, z_down_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    // random_init_array((uint8_t*)w_down_proj, 4096 * 14336 / 2);
    // random_init_array((uint8_t*)s_down_proj, 4096 * 14336 / QBLOCK_SIZE);
    // random_init_array((uint8_t*)z_down_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    // SVMUNMAP(queue, w_down_proj);
    // SVMUNMAP(queue, s_down_proj);
    // SVMUNMAP(queue, z_down_proj);

    // global in-out
    float* x = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * sizeof(float), 64);
    float* y = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * sizeof(float), 64);

    // scratch space
    int8_t* xq = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336, 64);
    float* xq_s = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 / QBLOCK_SIZE * sizeof(float), 64);
    float* s1 = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * sizeof(float), 64);
    float* s2 = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * sizeof(float), 64);

    // SVMMAP(queue, x, 4096 * sizeof(float));
    // SVMMAP(queue, y, 4096 * sizeof(float));
    // SVMMAP(queue, xq, 14336);
    // SVMMAP(queue, xq_s, 14336 / QBLOCK_SIZE * sizeof(float));
    // SVMMAP(queue, s1, 14336 * sizeof(float));
    // SVMMAP(queue, s2, 14336 * sizeof(float));
    // random_init_array((uint8_t*)x, 4096);
    // for (int i = 0; i < 14336; i++) {
    //     y[i % 4096] = 0.0f;
    //     s1[i] = 0.0f;
    //     s2[i] = 0.0f;
    //     xq[i] = 0;
    //     xq_s[i / QBLOCK_SIZE] = 0.0f;
    // }
    // SVMUNMAP(queue, x);
    // SVMUNMAP(queue, y);
    // SVMUNMAP(queue, xq);
    // SVMUNMAP(queue, xq_s);
    // SVMUNMAP(queue, s1);
    // SVMUNMAP(queue, s2);

    // ==== bench ====
    const int NIT = 200;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        // Q(x) --> xq
        f32_qi8f32s(x, xq, xq_s, 4096, 4);

        // up_proj @ xq --> s1
        q4f32s_qi8f32s_egemv(
            w_up_proj, s_up_proj, z_up_proj,
            xq, xq_s,
            s1,
            14336, 4096,
            4);

        // gate_proj @ xq --> s2
        q4f32s_qi8f32s_egemv(
            w_gate_proj, s_gate_proj, z_gate_proj,
            xq, xq_s,
            s2,
            14336, 4096,
            4);

        // Q(s2) --> xq
        f32_qi8f32s(s2, xq, xq_s, 14336, 4);

        // down_proj @ up_proj_out
        q4f32s_qi8f32s_egemv(
            w_down_proj, s_down_proj, z_down_proj,
            xq, xq_s,
            y,
            4096, 14336,
            4);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = 4096 * 14336 * 6 * (uint64_t)NIT;
    double flops_per_sec = flops_processed / sec;
    cout << "GFLOPS: " << flops_per_sec * 1e-9 << endl;
    double BANDWIDTH = (double)(4096 * 14336 * 3) * (4.28125 / 8) * (double)NIT / sec * 1e-9;
    cout << "BANDWIDTH: GB/s: " << BANDWIDTH << endl;
    cout << endl;

    // ==== cleanup ====
    clSVMFree(context, w_up_proj);
    clSVMFree(context, s_up_proj);
    clSVMFree(context, z_up_proj);
    clSVMFree(context, w_gate_proj);
    clSVMFree(context, s_gate_proj);
    clSVMFree(context, z_gate_proj);
    clSVMFree(context, w_down_proj);
    clSVMFree(context, s_down_proj);
    clSVMFree(context, z_down_proj);
    clSVMFree(context, x);
    clSVMFree(context, y);
    clSVMFree(context, xq);
    clSVMFree(context, xq_s);
    clSVMFree(context, s1);
    clSVMFree(context, s2);
}

int main(int argc, char** argv)
{
    cl_int clStatus;
    cl_device_id device[1];
    cl_uint numDevices = 1;
    clStatus = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, numDevices, device, NULL);
    CL_CHECK(clStatus, "clGetDeviceIDs");

    char dev_name[128];
    clStatus = clGetDeviceInfo(device[0], CL_DEVICE_NAME, 128, dev_name, nullptr);
    CL_CHECK(clStatus, "clGetDeviceInfo - CL_DEVICE_NAME");
    cout << "Chose Device: " << dev_name << endl;

    cl_uint max_compute_units;
    clStatus = clGetDeviceInfo(device[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, nullptr);
    CL_CHECK(clStatus, "clGetDeviceInfo - CL_DEVICE_MAX_COMPUTE_UNITS");
    cout << "Processing Elements: " << max_compute_units << endl;

    cl_ulong shared_memory_size;
    clStatus = clGetDeviceInfo(device[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &shared_memory_size, nullptr);
    CL_CHECK(clStatus, "clGetDeviceInfo - CL_DEVICE_LOCAL_MEM_SIZE");
    cout << "Shared Memory Size: " << shared_memory_size << " bytes" << endl;

    context = clCreateContext(NULL, 1, device, NULL, NULL, &clStatus);
    CL_CHECK(clStatus, "clCreateContext");

    cl_queue_properties queue_props[] = { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR, 0 };
    queue = clCreateCommandQueueWithProperties(context, device[0], queue_props, &clStatus);
    CL_CHECK(clStatus, "clCreateCommandQueue");

    // build kernels
    char* c_str_cl_src = (char*)cl_src.c_str();
    cl_program p = clCreateProgramWithSource(context, 1, (const char**)&c_str_cl_src, NULL, &clStatus);
    CL_CHECK(clStatus, "clCreateProgramWithSource");
#ifdef BENCH
    string CLC_FLAGS = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
#else
    string CLC_FLAGS = "-cl-std=CL2.0 -cl-mad-enable"; // -cl-fast-relaxed-math";
#endif

    clStatus = clBuildProgram(p, 1, device, CLC_FLAGS.c_str(), NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(p, device[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(p, device[0], CL_PROGRAM_BUILD_LOG, log_size + 1, log, NULL);
        cout << "Build Errors:\n\n"
             << log << endl;
        free(log);
        exit(1);
    }

    vec_add_kernel = clCreateKernel(p, "vec_add", &clStatus);
    CL_CHECK(clStatus, "clCreateKernel - vec_add");
    q4f32s_qi8f32s_egemv_kernel = clCreateKernel(p, "q4f32s_qi8f32s_egemv_kernel", &clStatus);
    CL_CHECK(clStatus, "clCreateKernel - q4f32s_qi8f32s_egemv_kernel");
    f32_qi8f32s_kernel = clCreateKernel(p, "f32_qi8f32s", &clStatus);
    CL_CHECK(clStatus, "clCreateKernel - f32s_qi8f32s");

    srand(1);
#ifndef BENCH
    testing::InitGoogleTest(&argc, argv);
    int test_result = RUN_ALL_TESTS();
#else
    bench_llama_ffn();
#endif

    // cleanup
    clReleaseKernel(vec_add_kernel);
    clReleaseKernel(q4f32s_qi8f32s_egemv_kernel);
    clReleaseKernel(f32_qi8f32s_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
#ifndef BENCH
    return test_result;
#endif
}

/*
void test_512x512_input()
{
    const int m = 512;
    const int n = 512;

    uint8_t* w = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / 2, 64);
    float* s = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(float), 64);
    float* in_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n / QBLOCK_SIZE * sizeof(float), 64);
    int8_t* out = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m, 64);
    float* out_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, m / QBLOCK_SIZE * sizeof(float), 64);

    // 1 - Trivial
    {
        SVMMAP(queue, w, m * n / 2);
        BROADCAST(w, m * n / 2, 0x55);
        SVMUNMAP(queue, w);

        SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
        BROADCAST(s, m * n / QBLOCK_SIZE, 2.0f);
        SVMUNMAP(queue, s);

        SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
        BROADCAST(z, m * n / QBLOCK_SIZE / 2, 0x11);
        SVMUNMAP(queue, z);

        SVMMAP(queue, in, n);
        BROADCAST(in, n, 2);
        SVMUNMAP(queue, in);

        SVMMAP(queue, in_scales, n / QBLOCK_SIZE);
        BROADCAST(in_scales, n / QBLOCK_SIZE, 1.0f);
        SVMUNMAP(queue, in_scales);

        SVMMAP(queue, out, m);
        memset(out, 0, m);
        SVMUNMAP(queue, out);

        SVMMAP(queue, out_scales, m / QBLOCK_SIZE);
        BROADCAST(out_scales, m / QBLOCK_SIZE, 81.92f);
        SVMUNMAP(queue, out_scales);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        SVMMAP(queue, out, m);
        bool passed = true;
        for (int i = 0; i < m; i++) {
            if (out[i] != 100) {
                cout << "Error: out[" << i << "] = " << (int)out[i] << endl;
                passed = false;
            }
        }
        SVMUNMAP(queue, out);
        if (passed) {
            cout << "Test 1 Passed!" << endl;
        } else {
            cout << "Test 1 Failed!" << endl;
        }
    }

    // 2 - unique weight scales, ensure proper loading
    {
        SVMMAP(queue, w, m * n / 2);
        BROADCAST(w, m * n / 2, 0x55);
        SVMUNMAP(queue, w);

        SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
        for (int out_qblock = 0; out_qblock < m / QBLOCK_SIZE; out_qblock++) {
            for (int in_qblock = 0; in_qblock < n / QBLOCK_SIZE; in_qblock++) {
                int qblock_id = out_qblock * n / QBLOCK_SIZE + in_qblock;
                for (int el = 0; el < QBLOCK_SIZE; el++) {
                    s[qblock_id * QBLOCK_SIZE + el] = (float)(el + 1);
                }
            }
        }
        SVMUNMAP(queue, s);

        SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
        BROADCAST(z, m * n / QBLOCK_SIZE / 2, 0x11);
        SVMUNMAP(queue, z);

        SVMMAP(queue, in, n);
        BROADCAST(in, n, 2);
        SVMUNMAP(queue, in);

        SVMMAP(queue, in_scales, n / QBLOCK_SIZE);
        BROADCAST(in_scales, n / QBLOCK_SIZE, 1.0f);
        SVMUNMAP(queue, in_scales);

        SVMMAP(queue, out, m);
        memset(out, 0, m);
        SVMUNMAP(queue, out);

        SVMMAP(queue, out_scales, m / QBLOCK_SIZE);
        BROADCAST(out_scales, m / QBLOCK_SIZE, 5242.88f);
        SVMUNMAP(queue, out_scales);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        SVMMAP(queue, out, m);
        bool passed = true;
        for (int i = 0; i < m; i++) {
            if (out[i] != round((i % QBLOCK_SIZE + 1) * 4096 / 5242.88f)) {
                cout << "Error: out[" << i << "] = " << (int)out[i] << endl;
                passed = false;
            }
        }
        SVMUNMAP(queue, out);
        if (passed) {
            cout << "Test 2 Passed!" << endl;
        } else {
            cout << "Test 2 Failed!" << endl;
        }
    }

    // 3 - trivial, but different input scales
    {
        SVMMAP(queue, w, m * n / 2);
        BROADCAST(w, m * n / 2, 0x55);
        SVMUNMAP(queue, w);

        SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
        BROADCAST(s, m * n / QBLOCK_SIZE, 2.0f);
        SVMUNMAP(queue, s);

        SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
        BROADCAST(z, m * n / QBLOCK_SIZE / 2, 0x11);
        SVMUNMAP(queue, z);

        SVMMAP(queue, in, n);
        BROADCAST(in, n, 2);
        SVMUNMAP(queue, in);

        SVMMAP(queue, in_scales, n / QBLOCK_SIZE);
        for (int i = 0; i < n / QBLOCK_SIZE; i++) {
            in_scales[i] = (float)(i + 1);
        }
        SVMUNMAP(queue, in_scales);

        SVMMAP(queue, out, m);
        memset(out, 0, m);
        SVMUNMAP(queue, out);

        SVMMAP(queue, out_scales, m / QBLOCK_SIZE);
        BROADCAST(out_scales, m / QBLOCK_SIZE, 204.8f);
        SVMUNMAP(queue, out_scales);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        SVMMAP(queue, out, m);
        bool passed = true;
        for (int i = 0; i < m; i++) {
            if (out[i] != 100) {
                cout << "Error: out[" << i << "] = " << (int)out[i] << endl;
                passed = false;
            }
        }
        SVMUNMAP(queue, out);
        if (passed) {
            cout << "Test 3 Passed!" << endl;
        } else {
            cout << "Test 3 Failed!" << endl;
        }
    }

    // 4 trivial - but 0 values after adjustment
    {
        SVMMAP(queue, w, m * n / 2);
        BROADCAST(w, m * n / 2, 0x00);
        SVMUNMAP(queue, w);

        SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
        BROADCAST(s, m * n / QBLOCK_SIZE, 2.0f);
        SVMUNMAP(queue, s);

        SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
        BROADCAST(z, m * n / QBLOCK_SIZE / 2, 0x44);
        SVMUNMAP(queue, z);

        SVMMAP(queue, in, n);
        BROADCAST(in, n, 2);
        SVMUNMAP(queue, in);

        SVMMAP(queue, in_scales, n / QBLOCK_SIZE);
        BROADCAST(in_scales, n / QBLOCK_SIZE, 1.0f);
        SVMUNMAP(queue, in_scales);

        SVMMAP(queue, out, m);
        memset(out, 0, m);
        SVMUNMAP(queue, out);

        SVMMAP(queue, out_scales, m / QBLOCK_SIZE);
        BROADCAST(out_scales, m / QBLOCK_SIZE, 81.92f);
        SVMUNMAP(queue, out_scales);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        SVMMAP(queue, out, m);
        bool passed = true;
        for (int i = 0; i < m; i++) {
            if (out[i] != -100) {
                cout << "Error: out[" << i << "] = " << (int)out[i] << endl;
                passed = false;
            }
        }
        SVMUNMAP(queue, out);
        if (passed) {
            cout << "Test 4 Passed!" << endl;
        } else {
            cout << "Test 4 Failed!" << endl;
        }
    }

    // 5 - Trivial, but the out scale is too small to prevent int8 overflow
    {
        SVMMAP(queue, w, m * n / 2);
        BROADCAST(w, m * n / 2, 0x00);
        SVMUNMAP(queue, w);

        SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
        BROADCAST(s, m * n / QBLOCK_SIZE, 2.0f);
        SVMUNMAP(queue, s);

        SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
        BROADCAST(z, m * n / QBLOCK_SIZE / 2, 0x44);
        SVMUNMAP(queue, z);

        SVMMAP(queue, in, n);
        BROADCAST(in, n, 2);
        SVMUNMAP(queue, in);

        SVMMAP(queue, in_scales, n / QBLOCK_SIZE);
        BROADCAST(in_scales, n / QBLOCK_SIZE, 1.0f);
        SVMUNMAP(queue, in_scales);

        SVMMAP(queue, out, m);
        memset(out, 0, m);
        SVMUNMAP(queue, out);

        SVMMAP(queue, out_scales, m / QBLOCK_SIZE);
        BROADCAST(out_scales, m / QBLOCK_SIZE, 1.0f);
        SVMUNMAP(queue, out_scales);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        SVMMAP(queue, out, m);
        bool passed = true;
        for (int i = 0; i < m; i++) {
            if (out[i] != -128) {
                cout << "Error: out[" << i << "] = " << (int)out[i] << endl;
                passed = false;
            }
        }
        SVMUNMAP(queue, out);
        if (passed) {
            cout << "Test 5 Passed!" << endl;
        } else {
            cout << "Test 5 Failed!" << endl;
        }
    }

    // 6 - alternating weights along the output dimension
    {
        SVMMAP(queue, w, m * n / 2);
        for (int row = 0; row < m; row++) { // row idx
            for (int col = 0; col < n / 2; col++) { // col idx
                w[row * n / 2 + col] = (row % 2 == 0) ? 0x33 : 0x55;
            }
        }
        SVMUNMAP(queue, w);

        SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
        BROADCAST(s, m * n / QBLOCK_SIZE, 2.0f);
        SVMUNMAP(queue, s);

        SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
        BROADCAST(z, m * n / QBLOCK_SIZE / 2, 0x11);
        SVMUNMAP(queue, z);

        SVMMAP(queue, in, n);
        BROADCAST(in, n, 2);
        SVMUNMAP(queue, in);

        SVMMAP(queue, in_scales, n / QBLOCK_SIZE);
        BROADCAST(in_scales, n / QBLOCK_SIZE, 1.0f);
        SVMUNMAP(queue, in_scales);

        SVMMAP(queue, out, m);
        memset(out, 0, m);
        SVMUNMAP(queue, out);

        SVMMAP(queue, out_scales, m / QBLOCK_SIZE);
        BROADCAST(out_scales, m / QBLOCK_SIZE, 81.92f);
        SVMUNMAP(queue, out_scales);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        SVMMAP(queue, out, m);
        bool passed = true;
        for (int i = 0; i < m; i++) {
            if (out[i] != (i % 2 == 0 ? 50 : 100)) {
                cout << "Error: out[" << i << "] = " << (int)out[i] << endl;
                passed = false;
            }
        }
        SVMUNMAP(queue, out);

        if (passed) {
            cout << "Test 6 Passed!" << endl;
        } else {
            cout << "Test 6 Failed!" << endl;
        }
    }

    // 7 - alternating zeros along out dim
    {
        SVMMAP(queue, w, m * n / 2);
        BROADCAST(w, m * n / 2, 0x11);
        SVMUNMAP(queue, w);

        SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
        BROADCAST(s, m * n / QBLOCK_SIZE, 2.0f);
        SVMUNMAP(queue, s);

        SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
        for (int out_qblock = 0; out_qblock < m / QBLOCK_SIZE; out_qblock++) {
            for (int in_qblock = 0; in_qblock < n / QBLOCK_SIZE; in_qblock++) {
                int qblock_id = out_qblock * n / QBLOCK_SIZE + in_qblock;
                for (int el = 0; el < QBLOCK_SIZE; el++) {
                    z[qblock_id * QBLOCK_SIZE / 2 + el] = (el % 2 == 0) ? 0x11 : 0x33;
                }
            }
        }
        SVMUNMAP(queue, z);

        SVMMAP(queue, in, n);
        BROADCAST(in, n, 2);
        SVMUNMAP(queue, in);

        SVMMAP(queue, in_scales, n / QBLOCK_SIZE);
        BROADCAST(in_scales, n / QBLOCK_SIZE, 1.0f);
        SVMUNMAP(queue, in_scales);

        SVMMAP(queue, out, m);
        memset(out, 0, m);
        SVMUNMAP(queue, out);

        SVMMAP(queue, out_scales, m / QBLOCK_SIZE);
        BROADCAST(out_scales, m / QBLOCK_SIZE, 40.96f);
        SVMUNMAP(queue, out_scales);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        SVMMAP(queue, out, m);
        bool passed = true;
        for (int i = 0; i < m; i += 4) {
            if (out[i] != 0 || out[i + 1] != 0 || out[i + 2] != -100 || out[i + 3] != -100) {
                cout << "Error: out[" << i << "] = " << (int)out[i] << endl;
                cout << "Error: out[" << i + 1 << "] = " << (int)out[i + 1] << endl;
                cout << "Error: out[" << i + 2 << "] = " << (int)out[i + 2] << endl;
                cout << "Error: out[" << i + 3 << "] = " << (int)out[i + 3] << endl;
                passed = false;
            }
        }
        SVMUNMAP(queue, out);

        if (passed) {
            cout << "Test 7 Passed!" << endl;
        } else {
            cout << "Test 7 Failed!" << endl;
        }
    }

    clSVMFree(context, w);
    clSVMFree(context, s);
    clSVMFree(context, z);
    clSVMFree(context, in);
    clSVMFree(context, in_scales);
    clSVMFree(context, out);
    clSVMFree(context, out_scales);
}

void test_dim_fuzz()
{
    cout << "### Dimension Fuzzing ###" << endl;

    uint8_t* w = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 32768 * 32768 / 2, 64);
    float* s = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 32768 * 32768 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 32768 * 32768 / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 32768 * sizeof(float), 64);
    float* in_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 32768 / QBLOCK_SIZE * sizeof(float), 64);
    int8_t* out = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 32768, 64);
    float* out_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 32768 / QBLOCK_SIZE * sizeof(float), 64);

    for (int m = 512; m <= 32768; m += 512) {
        for (int n = 512; n <= 32768; n += 512) {

            SVMMAP(queue, w, m * n / 2);
            BROADCAST(w, m * n / 2, 0x55);
            SVMUNMAP(queue, w);

            SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
            BROADCAST(s, m * n / QBLOCK_SIZE, 2.0f);
            SVMUNMAP(queue, s);

            SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
            BROADCAST(z, m * n / QBLOCK_SIZE / 2, 0x11);
            SVMUNMAP(queue, z);

            SVMMAP(queue, in, n);
            BROADCAST(in, n, 2);
            SVMUNMAP(queue, in);

            SVMMAP(queue, in_scales, n / QBLOCK_SIZE);
            BROADCAST(in_scales, n / QBLOCK_SIZE, 1.0f);
            SVMUNMAP(queue, in_scales);

            SVMMAP(queue, out, m);
            memset(out, 0, m);
            SVMUNMAP(queue, out);

            SVMMAP(queue, out_scales, m / QBLOCK_SIZE);
            float _os = (float)(n * 2 * 2 * 4) / 100.0f;
            BROADCAST(out_scales, m / QBLOCK_SIZE, _os);
            SVMUNMAP(queue, out_scales);

            q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

            SVMMAP(queue, out, m);
            bool passed = true;
            for (int i = 0; i < m; i++) {
                if (out[i] != 100) {
                    cout << "Error: out[" << i << "] = " << (int)out[i] << endl;
                    passed = false;
                }
            }
            SVMUNMAP(queue, out);

            if (!passed) {
                cout << "Fuzz Test 1 Failed for dim: " << m << ", " << n << endl;
                cout << "Output scale: " << _os << endl;
                cout << "Expected: " << 100 << endl;
                exit(0);
            }
        }
        cout << "Fuzz Test 1 Passed for dim: " << m << ", ?" << endl;
    }

    clSVMFree(context, w);
    clSVMFree(context, s);
    clSVMFree(context, z);
    clSVMFree(context, in);
    clSVMFree(context, in_scales);
    clSVMFree(context, out);
    clSVMFree(context, out_scales);
}

void random_init_array(char* arr, int len)
{
    for (int i = 0; i < len; i++) {
        arr[i] = rand() % 256;
    }
}

void bench_llama_up_proj()
{
    cout << "Benchmarking LLAMA Up Proj ..." << endl;
    cout << "Hidden Dim: 14336, Dim: 4096" << endl;
    cout << endl;

    int m = 14336;
    int n = 4096;

    uint8_t* w = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / 2, 64);
    float* s = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m * n / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, n, 64);
    float* input_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n / QBLOCK_SIZE * sizeof(float), 64);
    int8_t* out = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m, 64);
    float* output_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, m / QBLOCK_SIZE * sizeof(float), 64);
    SVMMAP(queue, w, m * n / 2);
    SVMMAP(queue, s, m * n / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, z, m * n / QBLOCK_SIZE / 2);
    SVMMAP(queue, in, n);
    SVMMAP(queue, input_scales, n / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)w, m * n / 2);
    random_init_array((char*)s, m * n / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)z, m * n / QBLOCK_SIZE / 2);
    random_init_array((char*)in, n);
    random_init_array((char*)input_scales, n / QBLOCK_SIZE * sizeof(float));
    SVMUNMAP(queue, w);
    SVMUNMAP(queue, s);
    SVMUNMAP(queue, z);
    SVMUNMAP(queue, in);
    SVMUNMAP(queue, input_scales);

    // ==== bench ====
    const int NIT = 200;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        q4f32s_qi8f32s_egemv_offline(w, s, z, in, input_scales, out, output_scales, m, n);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = 4096 * 14336 * 2 * (uint64_t)NIT;
    double flops_per_sec = flops_processed / sec;
    cout << "GFLOPS: " << flops_per_sec / 1e9 << endl;
    cout << endl;

    clSVMFree(context, w);
    clSVMFree(context, s);
    clSVMFree(context, z);
    clSVMFree(context, in);
    clSVMFree(context, input_scales);
    clSVMFree(context, out);
    clSVMFree(context, output_scales);
}

void bench_llama_ffn()
{
    // down proj 4096x14336
    // gate_proj 14336x4096
    // up_proj 14336x4096
    // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
    // we do not do the elementwise multiply
    // we do not apply SwiGLU non-linearity in the hidden_dim
    // just the matmuls. skips 14k ops out of 176M total (4096 * 14336 * 3)
    cout << "Benchmarking LLAMA FFN ..." << endl;
    cout << "down_proj @ (up_proj @ x * gate_proj @ x)" << endl;
    cout << "Hidden Dim: 14436, Dim: 4096" << endl;
    cout << endl;

    // ==== activations ====
    int8_t* io = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096, 64);
    float* input_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 / QBLOCK_SIZE * sizeof(float), 64);
    float* output_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 / QBLOCK_SIZE * sizeof(float), 64);
    SVMMAP(queue, io, 4096);
    SVMMAP(queue, input_scales, 4096 / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, output_scales, 4096 / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)io, 4096);
    random_init_array((char*)input_scales, 4096 / QBLOCK_SIZE);
    random_init_array((char*)output_scales, 4096 / QBLOCK_SIZE);
    SVMUNMAP(queue, io);
    SVMUNMAP(queue, input_scales);
    SVMUNMAP(queue, output_scales);

    // ==== up_proj ====
    uint8_t* w_up_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / 2, 64);
    float* s_up_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_up_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / QBLOCK_SIZE / 2, 64);
    int8_t* out_up_proj = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336, 64);
    float* out_scales_up_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 / QBLOCK_SIZE * sizeof(float), 64);
    SVMMAP(queue, w_up_proj, 14336 * 4096 / 2);
    SVMMAP(queue, s_up_proj, 14336 * 4096 / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, z_up_proj, 14336 * 4096 / QBLOCK_SIZE / 2);
    SVMMAP(queue, out_up_proj, 14336);
    SVMMAP(queue, out_scales_up_proj, 14336 / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)w_up_proj, 14336 * 4096 / 2);
    random_init_array((char*)s_up_proj, 14336 * 4096 / QBLOCK_SIZE);
    random_init_array((char*)z_up_proj, 14336 * 4096 / QBLOCK_SIZE / 2);
    random_init_array((char*)out_up_proj, 14336);
    random_init_array((char*)out_scales_up_proj, 14336 / QBLOCK_SIZE);
    SVMUNMAP(queue, w_up_proj);
    SVMUNMAP(queue, s_up_proj);
    SVMUNMAP(queue, z_up_proj);
    SVMUNMAP(queue, out_up_proj);
    SVMUNMAP(queue, out_scales_up_proj);

    // ==== gate_proj ====
    uint8_t* w_gate_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / 2, 64);
    float* s_gate_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_gate_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE / 2, 64);
    int8_t* out_gate_proj = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336, 64);
    float* out_scales_gate_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 / QBLOCK_SIZE * sizeof(float), 64);
    SVMMAP(queue, w_gate_proj, 4096 * 14336 / 2);
    SVMMAP(queue, s_gate_proj, 4096 * 14336 / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, z_gate_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    SVMMAP(queue, out_gate_proj, 14336);
    SVMMAP(queue, out_scales_gate_proj, 14336 / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)w_gate_proj, 4096 * 14336 / 2);
    random_init_array((char*)s_gate_proj, 4096 * 14336 / QBLOCK_SIZE);
    random_init_array((char*)z_gate_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    random_init_array((char*)out_gate_proj, 14336);
    random_init_array((char*)out_scales_gate_proj, 14336 / QBLOCK_SIZE);
    SVMUNMAP(queue, w_gate_proj);
    SVMUNMAP(queue, s_gate_proj);
    SVMUNMAP(queue, z_gate_proj);
    SVMUNMAP(queue, out_gate_proj);
    SVMUNMAP(queue, out_scales_gate_proj);

    // ==== down_proj ====
    uint8_t* w_down_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / 2, 64);
    float* s_down_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_down_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE / 2, 64);
    SVMMAP(queue, w_down_proj, 4096 * 14336 / 2);
    SVMMAP(queue, s_down_proj, 4096 * 14336 / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, z_down_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    random_init_array((char*)w_down_proj, 4096 * 14336 / 2);
    random_init_array((char*)s_down_proj, 4096 * 14336 / QBLOCK_SIZE);
    random_init_array((char*)z_down_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    SVMUNMAP(queue, w_down_proj);
    SVMUNMAP(queue, s_down_proj);
    SVMUNMAP(queue, z_down_proj);

    // ==== bench ====
    const int NIT = 200;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
        // up_proj @ x
        q4f32s_qi8f32s_egemv_offline(
            w_up_proj, s_up_proj, z_up_proj,
            io, input_scales,
            out_up_proj, out_scales_up_proj,
            14336, 4096);

        // gate_proj @ x
        q4f32s_qi8f32s_egemv_offline(
            w_gate_proj, s_gate_proj, z_gate_proj,
            io, input_scales,
            out_gate_proj, out_scales_gate_proj,
            14336, 4096);

        // down_proj @ up_proj_out
        q4f32s_qi8f32s_egemv_offline(
            w_down_proj, s_down_proj, z_down_proj,
            out_up_proj, out_scales_up_proj,
            io, output_scales,
            4096, 14336);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = 4096 * 14336 * 6 * (uint64_t)NIT;
    double flops_per_sec = flops_processed / sec;
    cout << "GFLOPS: " << flops_per_sec / 1e9 << endl;
    cout << endl;

    // ==== cleanup ====
    clSVMFree(context, io);
    clSVMFree(context, input_scales);
    clSVMFree(context, w_up_proj);
    clSVMFree(context, s_up_proj);
    clSVMFree(context, z_up_proj);
    clSVMFree(context, out_up_proj);
    clSVMFree(context, out_scales_up_proj);
    clSVMFree(context, w_gate_proj);
    clSVMFree(context, s_gate_proj);
    clSVMFree(context, z_gate_proj);
    clSVMFree(context, out_gate_proj);
    clSVMFree(context, out_scales_gate_proj);
    clSVMFree(context, w_down_proj);
    clSVMFree(context, s_down_proj);
    clSVMFree(context, z_down_proj);
}

void test_throttle()
{
    // down proj 4096x14336
    // gate_proj 14336x4096
    // up_proj 14336x4096
    // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
    // we do not do the elementwise multiply
    // we do not apply SwiGLU non-linearity in the hidden_dim
    // just the matmuls. skips 14k ops out of 176M total (4096 * 14336 * 3)
    cout << "Throttling iGPU with LLAMA FFN ..." << endl;
    cout << "down_proj @ (up_proj @ x * gate_proj @ x)" << endl;
    cout << "Hidden Dim: 14436, Dim: 4096" << endl;
    cout << "Simulating a ~500 token generation" << endl;
    cout << endl;

    // ==== activations ====
    int8_t* io = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096, 64);
    float* input_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 / QBLOCK_SIZE * sizeof(float), 64);
    float* output_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 / QBLOCK_SIZE * sizeof(float), 64);
    SVMMAP(queue, io, 4096);
    SVMMAP(queue, input_scales, 4096 / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, output_scales, 4096 / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)io, 4096);
    random_init_array((char*)input_scales, 4096 / QBLOCK_SIZE);
    random_init_array((char*)output_scales, 4096 / QBLOCK_SIZE);
    SVMUNMAP(queue, io);
    SVMUNMAP(queue, input_scales);
    SVMUNMAP(queue, output_scales);

    // ==== up_proj ====
    uint8_t* w_up_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / 2, 64);
    float* s_up_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_up_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * 4096 / QBLOCK_SIZE / 2, 64);
    int8_t* out_up_proj = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336, 64);
    float* out_scales_up_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 / QBLOCK_SIZE * sizeof(float), 64);
    SVMMAP(queue, w_up_proj, 14336 * 4096 / 2);
    SVMMAP(queue, s_up_proj, 14336 * 4096 / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, z_up_proj, 14336 * 4096 / QBLOCK_SIZE / 2);
    SVMMAP(queue, out_up_proj, 14336);
    SVMMAP(queue, out_scales_up_proj, 14336 / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)w_up_proj, 14336 * 4096 / 2);
    random_init_array((char*)s_up_proj, 14336 * 4096 / QBLOCK_SIZE);
    random_init_array((char*)z_up_proj, 14336 * 4096 / QBLOCK_SIZE / 2);
    random_init_array((char*)out_up_proj, 14336);
    random_init_array((char*)out_scales_up_proj, 14336 / QBLOCK_SIZE);
    SVMUNMAP(queue, w_up_proj);
    SVMUNMAP(queue, s_up_proj);
    SVMUNMAP(queue, z_up_proj);
    SVMUNMAP(queue, out_up_proj);
    SVMUNMAP(queue, out_scales_up_proj);

    // ==== gate_proj ====
    uint8_t* w_gate_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / 2, 64);
    float* s_gate_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_gate_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE / 2, 64);
    int8_t* out_gate_proj = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336, 64);
    float* out_scales_gate_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 / QBLOCK_SIZE * sizeof(float), 64);
    SVMMAP(queue, w_gate_proj, 4096 * 14336 / 2);
    SVMMAP(queue, s_gate_proj, 4096 * 14336 / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, z_gate_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    SVMMAP(queue, out_gate_proj, 14336);
    SVMMAP(queue, out_scales_gate_proj, 14336 / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)w_gate_proj, 4096 * 14336 / 2);
    random_init_array((char*)s_gate_proj, 4096 * 14336 / QBLOCK_SIZE);
    random_init_array((char*)z_gate_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    random_init_array((char*)out_gate_proj, 14336);
    random_init_array((char*)out_scales_gate_proj, 14336 / QBLOCK_SIZE);
    SVMUNMAP(queue, w_gate_proj);
    SVMUNMAP(queue, s_gate_proj);
    SVMUNMAP(queue, z_gate_proj);
    SVMUNMAP(queue, out_gate_proj);
    SVMUNMAP(queue, out_scales_gate_proj);

    // ==== down_proj ====
    uint8_t* w_down_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / 2, 64);
    float* s_down_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_down_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE / 2, 64);
    SVMMAP(queue, w_down_proj, 4096 * 14336 / 2);
    SVMMAP(queue, s_down_proj, 4096 * 14336 / QBLOCK_SIZE * sizeof(float));
    SVMMAP(queue, z_down_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    random_init_array((char*)w_down_proj, 4096 * 14336 / 2);
    random_init_array((char*)s_down_proj, 4096 * 14336 / QBLOCK_SIZE);
    random_init_array((char*)z_down_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    SVMUNMAP(queue, w_down_proj);
    SVMUNMAP(queue, s_down_proj);
    SVMUNMAP(queue, z_down_proj);

    // ==== bench ====
    const int NIT = 1600; // roughly 500 tokens
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
        // up_proj @ x
        q4f32s_qi8f32s_egemv_offline(
            w_up_proj, s_up_proj, z_up_proj,
            io, input_scales,
            out_up_proj, out_scales_up_proj,
            14336, 4096);

        // gate_proj @ x
        q4f32s_qi8f32s_egemv_offline(
            w_gate_proj, s_gate_proj, z_gate_proj,
            io, input_scales,
            out_gate_proj, out_scales_gate_proj,
            14336, 4096);

        // down_proj @ up_proj_out
        q4f32s_qi8f32s_egemv_offline(
            w_down_proj, s_down_proj, z_down_proj,
            out_up_proj, out_scales_up_proj,
            io, output_scales,
            4096, 14336);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = 4096 * 14336 * 6 * (uint64_t)NIT;
    double flops_per_sec = flops_processed / sec;
    cout << "GFLOPS: " << flops_per_sec / 1e9 << endl;
    cout << endl;

    // ==== cleanup ====
    clSVMFree(context, io);
    clSVMFree(context, input_scales);
    clSVMFree(context, w_up_proj);
    clSVMFree(context, s_up_proj);
    clSVMFree(context, z_up_proj);
    clSVMFree(context, out_up_proj);
    clSVMFree(context, out_scales_up_proj);
    clSVMFree(context, w_gate_proj);
    clSVMFree(context, s_gate_proj);
    clSVMFree(context, z_gate_proj);
    clSVMFree(context, out_gate_proj);
    clSVMFree(context, out_scales_gate_proj);
    clSVMFree(context, w_down_proj);
    clSVMFree(context, s_down_proj);
    clSVMFree(context, z_down_proj);
}*/
