#include "capstone/capstone.hpp"
#include "capstone/cl.hpp"
#include <cassert>
#include <iostream>

using namespace std;

static cl_program program;
static bool initialized = false;
// from `src/kernels/cl_src.zig`
extern const char* cl_src;

void cl_init(cl_context context, cl_device_id device)
{
    if (initialized) {
        return;
    }

    cl_int err;

    program = clCreateProgramWithSource(context, 1, &cl_src, nullptr, &err);
    CL_CHECK(err)
    err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t log_size;
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CL_CHECK(err)
        char* log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, log, NULL);
        CL_CHECK(err)
        cout << "Build Errors:\n\n"
             << log << endl;
        free(log);
        exit(1);
    }

    initialized = true;
}

cl_int cl_f32_qi8f32s(float* in, int8_t* out, float* out_s, int n, cl_command_queue queue, cl_event* ev)
{
    assert(128 <= n && n <= 32768 && "n must be between 128 and 32768");
    assert(n % 128 == 0 && "n must be a multiple of 128");

    cl_int clStatus;

    cl_kernel kernel = clCreateKernel(program, "f32_qi8f32s", &clStatus);
    CL_CHECK(clStatus);

    clStatus = clSetKernelArgSVMPointer(kernel, 0, in);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 1, out);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 2, out_s);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArg(kernel, 3, sizeof(int), &n);
    CL_CHECK(clStatus)

    // 4-256 elements per thread;
    int n_el_per_thread = ceil(n / 256); // 256 threads in this dimension
    while (QBLOCK_SIZE % n_el_per_thread != 0) {
        n_el_per_thread++;
    }
    clStatus = clSetKernelArg(kernel, 4, sizeof(int), &n_el_per_thread);
    CL_CHECK(clStatus)

    const size_t global_work_size = n / n_el_per_thread;
    const size_t local_work_size = QBLOCK_SIZE / n_el_per_thread;
    // local group size = (n/npt) / (QBLOCK_SIZE/npt) = n / QBLOCK_SIZE

    return clEnqueueNDRangeKernel(
        queue, kernel,
        1,
        nullptr, &global_work_size, &local_work_size,
        0, nullptr, ev);
}

cl_int vector_add(float* a, float* b, float* c, int n, cl_command_queue queue, cl_event* ev)
{
    cl_int clStatus;

    cl_kernel kernel = clCreateKernel(program, "vec_add", &clStatus);
    CL_CHECK(clStatus)

    clStatus = clSetKernelArgSVMPointer(kernel, 0, a);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 1, b);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 2, c);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArg(kernel, 3, sizeof(int), &n);
    CL_CHECK(clStatus)

    // there should be more checks like this for stability -> https://github.com/CNugteren/CLBlast/blob/master/src/routines/common.cpp
    cl_uint work_dim = 1;
    const size_t global_work_offset = 0;
    const size_t global_work_size = n;
    const size_t local_work_size = 1;
    return clEnqueueNDRangeKernel(
        queue, kernel,
        work_dim,
        &global_work_offset, &global_work_size, &local_work_size,
        0, nullptr, ev);
}

cl_int cl_q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_s,
    float* out,
    int m, int n,
    cl_command_queue queue,
    cl_event* ev)
{
    assert(m >= 512 && n >= 512 && "m and n must be at least 128");
    assert(m <= 32768 && n <= 32768 && "m and n can be at most 16384");
    assert(m % 512 == 0 && n % 512 == 0 && "m and n must be multiples of 128");

    cl_int clStatus;

    cl_kernel kernel = clCreateKernel(program, "q4f32s_qi8f32s_egemv_kernel", &clStatus);
    CL_CHECK(clStatus)

    clStatus = clSetKernelArgSVMPointer(kernel, 0, w);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 1, s);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 2, z);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 3, in);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 4, in_s);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArgSVMPointer(kernel, 5, out);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArg(kernel, 6, sizeof(int), &m);
    CL_CHECK(clStatus)
    clStatus = clSetKernelArg(kernel, 7, sizeof(int), &n);
    CL_CHECK(clStatus)
    int n_blocks_per_thread = n / 2 / 128;
    clStatus = clSetKernelArg(kernel, 8, sizeof(int), &n_blocks_per_thread);
    CL_CHECK(clStatus)

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
    return clEnqueueNDRangeKernel(
        queue, kernel,
        3, nullptr,
        global_work_size, local_work_size,
        0, nullptr, ev);
}

/*
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
*/
/*
int main(int argc, char** argv)
{
    cl_int clStatus;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform iris_graphics;
    for (auto& p : platforms) {
        cout << "Platform[" << p() << "] : " << p.getInfo<CL_PLATFORM_NAME>() << endl;
        if (p.getInfo<CL_PLATFORM_NAME>().find("Intel(R) OpenCL HD Graphics") != string::npos) {
            iris_graphics = p;
        }
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.size() == 0)
            continue;
        cout << "\t";
        for (auto& d : devices) {
            cout << "Device[" << d() << "] : " << d.getInfo<CL_DEVICE_NAME>() << ", ";
        }
        cout << endl;
    }
    if (iris_graphics() == 0) {
        cout << "Tests Search For 'Intel(R) OpenCL HD Graphics' OpenCL platform which was not found" << endl;
        exit(1);
    }

    cl_uint numDevices = 1;
    clStatus = clGetDeviceIDs(iris_graphics(), CL_DEVICE_TYPE_GPU, numDevices, &device, NULL);
    CL_CHECK(clStatus);

    char dev_name[128];
    clStatus = clGetDeviceInfo(device, CL_DEVICE_NAME, 128, dev_name, nullptr);
    CL_CHECK(clStatus);
    cout << "Chose Device: " << dev_name << endl;

    cl_uint max_compute_units;
    clStatus = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, nullptr);
    CL_CHECK(clStatus);
    cout << "Processing Elements: " << max_compute_units << endl;

    cl_ulong shared_memory_size;
    clStatus = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &shared_memory_size, nullptr);
    CL_CHECK(clStatus);
    cout << "Shared Memory Size: " << shared_memory_size << " bytes" << endl;

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &clStatus);
    CL_CHECK(clStatus);

    cl_queue_properties queue_props[] = { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, queue_props, &clStatus);
    CL_CHECK(clStatus);

    cl_init(context, device);

    srand(1);
    testing::InitGoogleTest(&argc, argv);
    int test_result = RUN_ALL_TESTS();

    // cleanup
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return test_result;
}
*/