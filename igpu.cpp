#include <CL/cl.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

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

#define CL_SRC(...) #__VA_ARGS__
const string cl_src = CL_SRC(

    __kernel void vec_add(__global const float* a, __global const float* b, __global float* c, const int n) {
        int i = get_global_id(0);
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }

    // m x n
    // global-x = m / 128
    // global-y = n / 128
    // local-x = 1
    // local-y = 4
    __kernel void q4f32s_qi8f32s_offline_v1(
        __global uchar4* restrict w,
        __global float* restrict s,
        __global uchar8* restrict z,
        __global char8* restrict in,
        __global float* restrict in_scales,
        __global char8* restrict out,
        __global float* restrict out_scales,
        int m, int n) {
        int blockIdx = get_global_id(0);
        int blockIdy = get_global_id(1);
        int threadIdx = get_local_id(0);
        int threadIdy = get_local_id(1);

        // num quant blocks
        int n_quant_blocks = get_local_size(0) / 128;

        // acc
        int4 acc = 0;

        // Set Zeros
        uchar4 tmp1 = w[0]; // figure out index
        uchar4 tmp2 = tmp1;
        tmp1 >>= 4;
        uchar8 zeros = { tmp1, tmp2 };
        zeros &= (uchar8)0x0F;

        for (int quant_block = 0; quant_block < n_quant_blocks; quant_block++) {
            for (int i = 0; i < 128; i += 8) {
                // Load Weights
                uchar4 tmp1 = w[0]; // figure out index
                uchar4 tmp2 = tmp1;
                tmp1 >>= 4;
                uchar8 weights = { tmp1, tmp2 };
                weights &= (uchar8)0x0F;

                weights -= zeros;

                short8 weights_short = convert_short8(weights);
                short8 input = convert_short8(in[0]); // figure out index
                short8 prod = weights_short * input;

                acc += convert_int4(prod.lo) + convert_int4(prod.hi);
            }
        }

        int acc_i32 = acc[0] + acc[1] + acc[2] + acc[3];

        // figure out index
        // out = atomic_fetch_add(out, acc);
    }

);

static cl_context context;
static cl_kernel vec_add_kernel;
static cl_command_queue queue;

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

    // test kernels
    cout << endl
         << "Testing vec_add..." << endl;
    test_vec_add();

    // cleanup
    clReleaseKernel(vec_add_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
