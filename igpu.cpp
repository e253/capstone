#include <CL/cl.h>
#include <cassert>
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

    __kernel __attribute__((vec_type_hint(short4))) void q4f32s_qi8f32s_offline_v1(
        __global uchar2* restrict w,
        __global float* restrict s,
        __global uchar* restrict z,
        __global char4* restrict in,
        __global float* restrict in_scales,
        __global char* restrict out,
        __global float* restrict out_scales,
        int m, int n, int n_blocks_per_thread) {
        const int QBLOCK_SIZE = 128;
        const int row_sz2_block = get_local_id(1);
        const int out_qblock = get_global_id(2);

        float acc1 = 0;
        float acc2 = 0;

        for (int qblock = 0; qblock < n_blocks_per_thread; qblock++) {
            // qblock-acc
            int2 acc1i = 0;
            int2 acc2i = 0;
            const int in_qblock = get_local_id(0) * n_blocks_per_thread + qblock;
            const int QBLOCK_ID = in_qblock * (n / QBLOCK_SIZE) + out_qblock;

            // Set Zeros
            uchar zero12 = z[QBLOCK_ID * QBLOCK_SIZE / 2 + row_sz2_block * 2 / 2];
            uchar tmp = zero12;
            uchar4 zero1 = (uchar4)((zero12 >> 4) & 0x0F);
            uchar4 zero2 = (uchar4)(tmp & 0x0F);

            for (int i = 0; i < 128; i += 4) {
                // Load Input
                short4 input = convert_short4(in[in_qblock * 128 / 4 + i / 4]); // `in` is char4*

                // Load Weights
                // block_offset = (in_qblock * 128)(row) * n/4(row_stride,uchar2,4 values) + (out_qblock * 128 / 4)(col,uchar2)
                // row_offset = row_sz2_block * 2 * n/4(row_stride,uchar2,4 values)
                uchar2 tmp1 = w[((in_qblock * 128 + row_sz2_block * 2) * n / 4 + (out_qblock * 128) / 4) + i / 4]; // check
                uchar2 tmp2 = tmp1;
                tmp1 >>= 4;
                uchar4 _weights1 = { tmp1, tmp2 };
                _weights1 &= (uchar4)0x0F;
                char4 weights1 = as_char4(_weights1); // safe becuase range is 0-15

                // same as weights1 but row offset is row_sz2_block * 2 * n/4(row_stride,uchar2) + 1
                uchar2 tmp3 = w[((in_qblock * 128 + row_sz2_block * 2 + 1) * n / 4 + (out_qblock * 128) / 4) + i / 4]; // check
                uchar2 tmp4 = tmp3;
                tmp3 >>= 4;
                uchar4 _weights2 = { tmp3, tmp4 };
                _weights2 &= (uchar4)0x0F;
                char4 weights2 = as_char4(_weights2); // safe becuase range is 0-15

                weights1 -= as_char4(zero1); // safe becuase range is 0-15
                weights2 -= as_char4(zero2);

                short4 prod = convert_short4(weights1) * input;
                short4 prod2 = convert_short4(weights2) * input;

                acc1i += convert_int2(prod.lo) + convert_int2(prod.hi);
                acc2i += convert_int2(prod2.lo) + convert_int2(prod2.hi);
            } // block process

            acc1 += (float)(acc1i.s0 + acc1i.s1) * s[QBLOCK_ID * QBLOCK_SIZE + row_sz2_block * 2] * in_scales[in_qblock]; // check s
            acc2 += (float)(acc2i.s0 + acc2i.s1) * s[QBLOCK_ID * QBLOCK_SIZE + row_sz2_block * 2 + 1] * in_scales[in_qblock]; // check s
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
    // wg_id = { 0, 0-_m/128, 0-_m/128 }
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
    float* in_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float), 64);
    int8_t* out = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, m, 64);
    float* out_scales = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float), 64);

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

    clSVMFree(context, w);
    clSVMFree(context, s);
    clSVMFree(context, z);
    clSVMFree(context, in);
    clSVMFree(context, in_scales);
    clSVMFree(context, out);
    clSVMFree(context, out_scales);
}

int main()
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

    // cl_queue_properties queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_ON_DEVICE };
    queue = clCreateCommandQueueWithProperties(context, device[0], nullptr, &clStatus);
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

    // cleanup
    clReleaseKernel(vec_add_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
