#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

#define QBLOCK_SIZE 128

#define CL_CHECK(status, op)                                            \
    if (status != CL_SUCCESS) {                                         \
        cout << "OpenCL error: " << status << " During " << op << endl; \
        exit(1);                                                        \
    }
#define SVMMAP(queue, svm_ptr, size) clEnqueueSVMMap(queue, CL_TRUE, CL_MEM_READ_WRITE, svm_ptr, size, 0, NULL, NULL)
#define SVMUNMAP(queue, svm_ptr) clEnqueueSVMUnmap(queue, svm_ptr, 0, NULL, NULL)
#define BROADCAST(ptr, len, v)    \
    for (int i = 0; i < len; i++) \
    ptr[i] = v

static cl_context context;
static cl_kernel vec_add_kernel;
static cl_kernel q4f32s_qi8f32s_offline_v1_kernel;
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

    __kernel void q4f32s_qi8f32s_offline_v1(
        __global uchar4* restrict w,
        __global float* restrict s,
        __global uchar* restrict z,
        __global char8* restrict in,
        __global float* restrict in_scales,
        __global char* restrict out,
        __global float* restrict out_scales,
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
                _acc1 = mad(convert_float(get1(weights1.s2) - zero1), convert_float(input.s3), _acc1);
                _acc1 = mad(convert_float(get0(weights1.s2) - zero1), convert_float(input.s4), _acc1);
                _acc1 = mad(convert_float(get1(weights1.s3) - zero1), convert_float(input.s5), _acc1);
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

            acc1 += (float)_acc1 * s[QBLOCK_ID * QBLOCK_SIZE + row_sz2_block * 2] * in_scales[in_qblock]; // check s
            acc2 += (float)_acc2 * s[QBLOCK_ID * QBLOCK_SIZE + row_sz2_block * 2 + 1] * in_scales[in_qblock]; // check s
        } // qblock

        __local float acc1_local[2][64];
        __local float acc2_local[2][64];
        acc1_local[get_local_id(0)][row_sz2_block] = acc1;
        acc2_local[get_local_id(0)][row_sz2_block] = acc2;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_local_id(0) == 0) {
            acc1 = acc1_local[0][row_sz2_block] + acc1_local[1][row_sz2_block];
            acc2 = acc2_local[0][row_sz2_block] + acc2_local[1][row_sz2_block];
            out[out_qblock * 128 + row_sz2_block * 2] = clamp(acc1 / out_scales[out_qblock]); // check out_scales
            out[out_qblock * 128 + row_sz2_block * 2 + 1] = clamp(acc2 / out_scales[out_qblock]); // check out_scales
        }
    });

// DO NOT CALL FROM MULTIPLE THREADS!
void q4f32s_qi8f32s_egemv_offline(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    int8_t* out,
    float* out_scales,
    int m, int n)
{
    assert(m >= 512 && n >= 512 && "m and n must be at least 128");
    assert(m <= 32768 && n <= 32768 && "m and n can be at most 16384");
    assert(m % 512 == 0 && n % 512 == 0 && "m and n must be multiples of 128");

    cl_int clStatus;
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_offline_v1_kernel, 0, w);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - w")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_offline_v1_kernel, 1, s);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - s")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_offline_v1_kernel, 2, z);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - z")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_offline_v1_kernel, 3, in);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - in")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_offline_v1_kernel, 4, in_scales);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - in_scales")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_offline_v1_kernel, 5, out);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - out")
    clStatus = clSetKernelArgSVMPointer(q4f32s_qi8f32s_offline_v1_kernel, 6, out_scales);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - out_scales")
    clStatus = clSetKernelArg(q4f32s_qi8f32s_offline_v1_kernel, 7, sizeof(int), &m);
    CL_CHECK(clStatus, "clSetKernelArg - m")
    clStatus = clSetKernelArg(q4f32s_qi8f32s_offline_v1_kernel, 8, sizeof(int), &n);
    CL_CHECK(clStatus, "clSetKernelArg - n")
    int n_blocks_per_thread = n / 2 / 128;
    clStatus = clSetKernelArg(q4f32s_qi8f32s_offline_v1_kernel, 9, sizeof(int), &n_blocks_per_thread);
    CL_CHECK(clStatus, "clSetKernelArg - n_blocks_per_thread")

    cl_event event;
    const size_t _m = m;
    const size_t _n = n;
    // global work size is the number of work items in each dimension
    const size_t global_work_size[] = { 2, 64, _m / 128 };
    // local work size is the number of work items in each work group
    const size_t local_work_size[] = { 2, 64, 1 };
    // n work groups per dimension is found by dividing the global work size by the local work size
    // in each dimension
    // local_id = { 0-1, 0-63, 0-m/128 - 1 }
    // each row is split in two halfs
    // each out_block is split into 64 threads doing 2 rows each.
    clStatus = clEnqueueNDRangeKernel(
        queue, q4f32s_qi8f32s_offline_v1_kernel,
        3, nullptr,
        global_work_size, local_work_size,
        0, nullptr, &event);
    CL_CHECK(clStatus, "q4f32s_qi8f32s_offline_v1_kernel invocation")

    clStatus = clWaitForEvents(1, &event);
    CL_CHECK(clStatus, "clWaitForEvents - q4f32s_qi8f32s_offline_v1_kernel")
    clStatus = clReleaseEvent(event);
    CL_CHECK(clStatus, "clReleaseEvent - q4f32s_qi8f32s_offline_v1_kernel")

    clStatus = clFinish(queue);
    CL_CHECK(clStatus, "clFinish - q4f32s_qi8f32s_offline_v1_kernel")
}

void q4f32s_qi8f32s_egemv_offline_ev(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    int8_t* out,
    float* out_scales,
    int m, int n,
    cl_uint n_events,
    const cl_event* event_wait_list,
    cl_event* event)
{
    assert(m >= 512 && n >= 512 && "m and n must be at least 128");
    assert(m <= 32768 && n <= 32768 && "m and n can be at most 16384");
    assert(m % 512 == 0 && n % 512 == 0 && "m and n must be multiples of 128");

    cl_int clStatus;
    cl_kernel _k = clCloneKernel(q4f32s_qi8f32s_offline_v1_kernel, &clStatus);

    clStatus = clSetKernelArgSVMPointer(_k, 0, w);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - w")
    clStatus = clSetKernelArgSVMPointer(_k, 1, s);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - s")
    clStatus = clSetKernelArgSVMPointer(_k, 2, z);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - z")
    clStatus = clSetKernelArgSVMPointer(_k, 3, in);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - in")
    clStatus = clSetKernelArgSVMPointer(_k, 4, in_scales);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - in_scales")
    clStatus = clSetKernelArgSVMPointer(_k, 5, out);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - out")
    clStatus = clSetKernelArgSVMPointer(_k, 6, out_scales);
    CL_CHECK(clStatus, "clSetKernelArgSVMPointer - out_scales")
    clStatus = clSetKernelArg(_k, 7, sizeof(int), &m);
    CL_CHECK(clStatus, "clSetKernelArg - m")
    clStatus = clSetKernelArg(_k, 8, sizeof(int), &n);
    CL_CHECK(clStatus, "clSetKernelArg - n")
    int n_blocks_per_thread = n / 2 / 128;
    clStatus = clSetKernelArg(_k, 9, sizeof(int), &n_blocks_per_thread);
    CL_CHECK(clStatus, "clSetKernelArg - n_blocks_per_thread")

    const size_t _m = m;
    const size_t _n = n;
    // global work size is the number of work items in each dimension
    const size_t global_work_size[] = { 2, 64, _m / 128 };
    // local work size is the number of work items in each work group
    const size_t local_work_size[] = { 2, 64, 1 };
    // n work groups per dimension is found by dividing the global work size by the local work size
    // in each dimension
    // local_id = { 0-1, 0-63, 0-m/128 - 1 }
    // each row is split in two halfs
    // each out_block is split into 64 threads doing 2 rows each.
    clStatus = clEnqueueNDRangeKernel(
        queue, q4f32s_qi8f32s_offline_v1_kernel,
        3, nullptr,
        global_work_size, local_work_size,
        n_events, event_wait_list, event);
    CL_CHECK(clStatus, "q4f32s_qi8f32s_offline_v1_kernel invocation")
    clStatus = clReleaseKernel(_k);
    CL_CHECK(clStatus, "clReleaseKernel - q4f32s_qi8f32s_offline_v1_kernel")
}

void test_vec_add()
{
    const int n = 1024;

    float* a = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(float), 64);
    float* b = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(float), 64);
    float* c = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, n * sizeof(float), 64);

    SVMMAP(queue, a, n * sizeof(float));
    BROADCAST(a, n, 1.0f);
    SVMUNMAP(queue, a);

    SVMMAP(queue, b, n * sizeof(float));
    BROADCAST(b, n, 2.0f);
    SVMUNMAP(queue, b);

    SVMMAP(queue, c, n * sizeof(float));
    BROADCAST(c, n, 0.0f);
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

    SVMMAP(queue, c, n * sizeof(float));
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != 3.0f) {
            cout << "Error: c[" << i << "] = " << c[i] << endl;
            passed = false;
        }
    }
    SVMUNMAP(queue, c);
    if (passed) {
        cout << "Vector Add Passed!" << endl;
    } else {
        cout << "Vector Add Failed!" << endl;
    }

    clSVMFree(context, a);
    clSVMFree(context, b);
    clSVMFree(context, c);
}

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
    string CLC_FLAGS = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
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
    q4f32s_qi8f32s_offline_v1_kernel = clCreateKernel(p, "q4f32s_qi8f32s_offline_v1", &clStatus);
    CL_CHECK(clStatus, "clCreateKernel - q4f32s_qi8f32s_offline_v1");

    // test kernels
    cout << endl
         << "Testing vec_add..." << endl;
    test_vec_add();

    cout << endl
         << "Testing 512x512 input..." << endl;
    test_512x512_input();

    if (argc == 2) {
        if (string(argv[1]) == "fuzz") {
            cout << "Fuzzing tests across all supported input dimensions ..." << endl;
            cout << "This will take a long time" << endl;
            test_dim_fuzz();
        } else if (string(argv[1]) == "throttle") {
            test_throttle();
            exit(0);
        }
    } else {
        cout << "Skipping Fuzz. run `./cpu.exe fuzz` to fuzz" << endl;
    }
    cout << endl;

    bench_llama_up_proj();
    bench_llama_ffn();

    // cleanup
    clReleaseKernel(vec_add_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
