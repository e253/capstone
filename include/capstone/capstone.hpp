#pragma once

#include "thread.hpp"
#include <cstdint>

#define QBLOCK_SIZE 128

// convert f32 to f16
#define F16(x) _cvtss_sh(x, 0)
// convert f16 to f32
#define F32(x) _cvtsh_ss(x)

void ref_f32_qi8f32s(float* in, int8_t* out, float* out_s, int n);
void ref_q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_s,
    float* out,
    int m, int n);

void f32_qi8f32s(float* in, int8_t* out, float* out_s, int n, int n_threads);
void q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_s,
    float* out,
    int m, int n,
    int n_threads);

struct f32_qi8f32s_params {
    float* in;
    int8_t* out;
    float* out_s;
    int n;
    int tid;
    int n_threads;
};

thread_ret_t f32_qi8f32s_thread(void* params);

struct q4f32s_qi8f32s_egemv_params {
    uint8_t* w;
    float* s;
    uint8_t* z;
    int8_t* in;
    float* in_s;
    float* out;
    int m;
    int n;
    int tid;
    int n_threads;
};

thread_ret_t q4f32s_qi8f32s_egemv_thread(void* params);

struct q4f32s_tensor {
    uint8_t* data;
    uint8_t* z;
    float* s;
    int m;
    int n;
};

struct f32_vector {
    float* data;
    int n;
};

struct qi8f32s_vector {
    int8_t* data;
    float* s;
    int n;
};

void q4f32s_qi8f32s_ffn(
    q4f32s_tensor* up_proj,
    q4f32s_tensor* gate_proj,
    q4f32s_tensor* down_proj,
    f32_vector* x, qi8f32s_vector* xq,
    f32_vector* y,
    f32_vector* s1, f32_vector* s2,
    int n_threads);
// Dequant(x) --> xq --> up_proj   --> F32 (scratch2)
//                   --> gate_proj --> F32 (scratch1) (abandoned)
//                                                   --> down_proj --> F32 (scratch1)

struct q4f32s_qi8f32s_ffn_params {
    q4f32s_tensor* up_proj;
    q4f32s_tensor* gate_proj;
    q4f32s_tensor* down_proj;
    f32_vector* x;
    qi8f32s_vector* xq;
    f32_vector* y;
    f32_vector* s1;
    f32_vector* s2;
    int n_threads;
    int tid;
    pthread_barrier_t* op_barrier;
};

thread_ret_t q4f32s_qi8f32s_ffn_thread(void* params);