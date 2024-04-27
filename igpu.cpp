#include <CL/cl.h>
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

    void atomicAddClamp(__global char* ptr, float val) {
        char old = *ptr;
        char next = convert_char_sat((float)old + val);
        while (atomic_cmpxchg(ptr, old, next) != old) {
            old = *ptr;
            next = convert_char_sat((float)old + val);
        }
    }

    // m x n
    // global-x = m / 128
    // global-y = n / 128
    // One work group per 128 x 128 block
    // local-x 0 <--> 64 (2 rows per thread)
    // local-y 0 <--> 0
    // Each work item computes 4 x 128
    __kernel void q4f32s_qi8f32s_offline_v1(
        __global uchar2* restrict w,
        __global float* restrict s,
        __global uchar* restrict z,
        __global char4* restrict in,
        __global float* restrict in_scales,
        __global char* restrict out,
        __global float* restrict out_scales,
        int m, int n) {
        const int QBLOCK_SIZE = 128;
        const int blockIdx = get_group_id(0);
        const int blockIdy = get_group_id(1);
        const int threadIdx = get_local_id(0);
        const int threadIdy = get_local_id(1);

        // printf("Thread (%d, %d) in Block (%d, %d) acknowledging\n", threadIdx, threadIdy, blockIdx, blockIdy);

        // acc
        int2 acc1 = 0;
        int2 acc2 = 0;

        int QBLOCK_ID = blockIdx * (n / QBLOCK_SIZE) + blockIdx;

        // Set Zeros
        uchar zero12 = z[QBLOCK_ID * QBLOCK_SIZE / 2 + threadIdx];
        uchar tmp = zero12;
        uchar4 zero1 = (uchar4)((zero12 >> 4) & 0x0F);
        uchar4 zero2 = (uchar4)(tmp & 0x0F);

        for (int i = 0; i < 128 / 4; i++) {
            // Load Input
            char4 cinput = in[blockIdy * 128 + i];
            short4 input = convert_short4(cinput);
            // printf("B(%d, %d):(%d, %d)[%d] - [%d, %d, %d, %d]\n", threadIdx, threadIdy, blockIdx, blockIdy, i, (int)input.s0, (int)input.s1, (int)input.s2, (int)input.s3);

            // Load Weights
            uchar2 tmp1 = w[(blockIdx * 128 * n / 2 + blockIdy * 128) + i];
            uchar2 tmp2 = tmp1;
            tmp1 >>= 4;
            uchar4 weights1 = { tmp1, tmp2 };
            weights1 &= (uchar4)0x0F;
            // printf("Weights1: (%d, %d) - [%d, %d, %d, %d]\n", threadIdx, threadIdy, (int)weights1.s0, (int)weights1.s1, (int)weights1.s2, (int)weights1.s3);

            uchar2 tmp3 = w[0]; // figure out index
            uchar2 tmp4 = tmp3;
            tmp3 >>= 4;
            uchar4 weights2 = { tmp3, tmp4 };
            weights2 &= (uchar4)0x0F;
            // printf("Weights2: (%d, %d) - [%d, %d, %d, %d]\n", threadIdx, threadIdy, (int)weights2.s0, (int)weights2.s1, (int)weights2.s2, (int)weights2.s3);

            weights1 -= zero1;
            weights2 -= zero2;
            // printf("Weights1 (zeroed): (%d, %d) - [%d, %d, %d, %d]\n", threadIdx, threadIdy, (int)weights1.s0, (int)weights1.s1, (int)weights1.s2, (int)weights1.s3);
            // printf("Weights2 (zeroed): (%d, %d) - [%d, %d, %d, %d]\n", threadIdx, threadIdy, (int)weights2.s0, (int)weights2.s1, (int)weights2.s2, (int)weights2.s3);

            short4 weights1_short = convert_short4(weights1);
            short4 prod = weights1_short * input;
            // printf("Prod: (%d, %d) - [%d, %d, %d, %d]\n", threadIdx, threadIdy, (int)prod.s0, (int)prod.s1, (int)prod.s2, (int)prod.s3);

            acc1 += convert_int2(prod.lo) + convert_int2(prod.hi);

            short4 weights2_short = convert_short4(weights2);
            short4 prod2 = weights2_short * input;
            // printf("Prod2: (%d, %d) - [%d, %d, %d, %d]\n", threadIdx, threadIdy, (int)prod.s0, (int)prod.s1, (int)prod.s2, (int)prod.s3);

            acc2 += convert_int2(prod2.lo) + convert_int2(prod2.hi);
        }
        // printf("Acc (final): (%d, %d) - [%d, %d]\n", threadIdx, threadIdy, acc1.s0, acc1.s1);
        // printf("Acc2 (final): (%d, %d) - [%d, %d]\n", threadIdx, threadIdy, acc2.s0, acc2.s1);

        // Expensive - fewer y blocks would be ideal
        float io_scale = in_scales[0] / out_scales[0];
        // printf("IO Scale: %f\n", io_scale);

        float scale1 = s[QBLOCK_ID + threadIdx * 2];
        // printf("(%d, %d) - Scale1: %f\n", threadIdx, threadIdy, scale1);
        float acc1_reduced = (float)(acc1.s0 + acc1.s1) * scale1 * io_scale;
        while (true) {
            char old = out[blockIdx * 128 + threadIdx * 2];
            char next = convert_char_sat((float)old + acc1_reduced);
            if (atomic_compare_exchange_strong(out + blockIdx * 128 + threadIdx * 2, &old, next)) {
                break;
            }
        }
        // atomicAddClamp(out + blockIdx * 128 + threadIdx * 2, acc1_reduced);
        //  char acc1_reduced_clamped = convert_char_sat(acc1_reduced); // clamp_i8(acc1_reduced);
        //   printf("Writing Back %d, for out[%d]\n", (int)acc1_reduced_clamped, blockIdx * 128 + threadIdx * 2);
        //   out[blockIdx * 128 + threadIdx * 2] = acc1_reduced_clamped;

        float scale2 = s[QBLOCK_ID + threadIdx * 2 + 1];
        // printf("(%d, %d) - Scale2: %f\n", threadIdx, threadIdy, scale2);
        float acc2_reduced = (float)(acc2.s0 + acc2.s1) * scale2 * io_scale;
        char acc2_reduced_clamped = convert_char_sat(acc2_reduced);
        // printf("Writing Back %d, for out[%d]\n", (int)acc2_reduced_clamped, blockIdx * 128 + threadIdx * 2 + 1);
        // out[blockIdx * 128 + threadIdx * 2 + 1] = acc2_reduced_clamped;
        atomicAddClamp(out + blockIdx * 128 + threadIdx * 2 + 1, acc2_reduced);
    }

);

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
    cl_int clStatus;
    cl_kernel _q4f32s_qi8f32s_offline_v1_kernel = clCloneKernel(q4f32s_qi8f32s_offline_v1_kernel, &clStatus);
    CL_CHECK(clStatus, "clCloneKernel")

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

    cl_event event;
    const size_t _m = m;
    const size_t _n = n;
    // global work size is the number of work items in each dimension
    const size_t global_work_size[] = { (_m / 128) * 64, _n / 128 };
    // local work size is the number of work items in each work group
    const size_t local_work_size[] = { 64, 1 };
    // n work groups per dimension is found by dividing the global work size by the local work size
    // in each dimension
    clStatus = clEnqueueNDRangeKernel(
        queue, q4f32s_qi8f32s_offline_v1_kernel,
        2, nullptr,
        global_work_size, local_work_size,
        0, nullptr, &event);
    CL_CHECK(clStatus, "q4f32s_qi8f32s_offline_v1_kernel invocation")

    clStatus = clWaitForEvents(1, &event);
    CL_CHECK(clStatus, "clWaitForEvents")
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

void test_128x128_input()
{
    const int m = 128;
    const int n = 128;

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

        SVMMAP(queue, in_scales, 1);
        in_scales[0] = 1.0f;
        SVMUNMAP(queue, in_scales);

        SVMMAP(queue, out, m);
        memset(out, 0, m);
        SVMUNMAP(queue, out);

        SVMMAP(queue, out_scales, 1);
        out_scales[0] = 20.48f;
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
         << "Testing 128x128 input..." << endl;
    test_128x128_input();

    // cleanup
    clReleaseKernel(vec_add_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
