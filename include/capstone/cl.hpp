#ifndef __CAPSTONE_CL_HPP
#define __CAPSTONE_CL_HPP

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#define CL_CHECK(status)                          \
    if (status != CL_SUCCESS) {                   \
        cout << "OpenCL error '" << status << "'" \
             << " in " << __FUNCTION__ << "()"    \
             << ", line " << __LINE__             \
             << endl;                             \
        exit(1);                                  \
    }

void cl_init(cl_context context, cl_device_id device);
cl_int vector_add(float* a, float* b, float* c, int n, cl_command_queue queue, cl_event* ev);
cl_int cl_f32_qi8f32s(float* in, int8_t* out, float* out_s, int n, cl_command_queue queue, cl_event* ev);
cl_int cl_q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_s,
    float* out,
    int m, int n,
    cl_command_queue queue,
    cl_event* ev);

#endif // __CAPSTONE_CL_HPP