#include "capstone/capstone.hpp"
#include "capstone/cl.hpp"
#include <chrono>
#include <iostream>

using namespace std;

int main()
{
    // ==== CL Loading ====
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform iris_graphics;
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
    cout << "Executing benchmarks on '" << iris_devices[0].getInfo<CL_DEVICE_NAME>() << "'" << endl;

    cl_int err;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    cl_queue_properties queue_props[] = { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    CL_CHECK(err);

    cl_init(context, device);

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

    // ==== gate_proj ====
    uint8_t* w_gate_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / 2, 64);
    float* s_gate_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_gate_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE / 2, 64);

    // ==== down_proj ====
    uint8_t* w_down_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / 2, 64);
    float* s_down_proj = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_down_proj = (uint8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * 14336 / QBLOCK_SIZE / 2, 64);

    // ==== global in-out ====
    float* x = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * sizeof(float), 64);
    float* y = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 4096 * sizeof(float), 64);

    // ==== scratch space ====
    int8_t* xq = (int8_t*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336, 64);
    float* xq_s = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 / QBLOCK_SIZE * sizeof(float), 64);
    float* s1 = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * sizeof(float), 64);
    float* s2 = (float*)clSVMAlloc(context, CL_MEM_READ_WRITE, 14336 * sizeof(float), 64);

    // ==== bench ====
    const int NIT = 200;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        cl_event evs[5];

        // Q(x) --> xq
        cl_int err = cl_f32_qi8f32s(x, xq, xq_s, 4096, queue, &evs[0]);
        CL_CHECK(err)

        // up_proj @ xq --> s1
        err = cl_q4f32s_qi8f32s_egemv(
            w_up_proj, s_up_proj, z_up_proj,
            xq, xq_s,
            s1,
            14336, 4096,
            queue, &evs[1]);
        CL_CHECK(err)

        // gate_proj @ xq --> s2
        err = cl_q4f32s_qi8f32s_egemv(
            w_gate_proj, s_gate_proj, z_gate_proj,
            xq, xq_s,
            s2,
            14336, 4096,
            queue, &evs[2]);
        CL_CHECK(err)

        // Q(s2) --> xq
        err = cl_f32_qi8f32s(s2, xq, xq_s, 14336, queue, &evs[3]);
        CL_CHECK(err)

        // down_proj @ up_proj_out
        err = cl_q4f32s_qi8f32s_egemv(
            w_down_proj, s_down_proj, z_down_proj,
            xq, xq_s,
            y,
            4096, 14336,
            queue, &evs[4]);
        CL_CHECK(err)

        clWaitForEvents(5, evs);
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
