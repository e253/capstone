#pragma once

#include <cstdint>

// nice potential api. leaving for now
/*
enum CapDataType {
    Q4_F32S, // 4 bit weights, f32 scales, 4 bit zeros
    Q4_F16S,
    Q8_F32S,
    Q8_F16S,
    F16,
    F32,
};

enum CapBackend {
    CPU,
    CLSVM, // shared virtual memory, allocated for igpu
};

struct captensor {
    void* data;
    void* scale;
    void* zeros;
    int ndim;
    int dim[4];
    DataType dtype;
    Backend backend;
};

captensor* cap_new_tensor_2d(int rows, int cols, CapDataType dtype, CapBackend backend);

void cap_egemv(captensor* A, captensor* x, captensor* y);
void cap_ffn(captensor* up_proj, captensor* gate_proj, captensor* down_proj, captensor* x, captensor* y);
*/

#define QBLOCK_SIZE 128

// convert f32 to f16
#define F16(x) _cvtss_sh(x, 0)
// convert f16 to f32
#define F32(x) _cvtsh_ss(x)

void ref_f32_qi8f32s(float* x0, int8_t* x1, float* x1_s, int n);
void ref_q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    float* out,
    int m, int n);

void f32_qi8f32s(float* x0, int8_t* x1, float* x1_s, int n);
void f32_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    float* out,
    int m, int n);