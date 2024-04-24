#pragma once

#include <cstdint>

#define QBLOCK_SIZE 128
// column major indexing helper
#define CM(ptr, row, col, col_stride) ((ptr) + (col * (col_stride)) + (row))

extern "C" {
// Call before using the ukernel
void q4f32s_ukernel_prelude();
// Call after using the ukernel to put result in ptr.
void q4f32s_ukernel_epiloque(float* ptr);

void q4f32s_ukernel(
    uint8_t* w, // Weight, offset from the global pointer
    uint64_t w_cs, // Col stride for weights
    float* scales, // Weight scales, offset from the global pointer
    uint64_t scales_cs, // Col stride for scales
    uint8_t* zeros, // Weight zeros, offset from the global pointer
    uint64_t zeros_cs, // Col stride for zeros
    float* in, // input, offset from the global pointer
    float* out, // out, offset from the global pointer
    uint64_t cols);

void q4f32s_128x128_ukernel(
    uint8_t* w,
    uint64_t w_rs,
    float* scales,
    uint8_t* zeros,
    float* in,
    float* out);

void q4f32s_q8in_128x128_ukernerl(
    uint8_t* w,
    uint64_t w_rs,
    float* scales,
    uint8_t* zeros,
    uint64_t zeros_cs,
    int8_t* in,
    float* out);

void q4f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    float* in,
    float* out,
    int m, int n);

void* _alloc(std::size_t size, std::size_t alignment);
void _free(void* ptr);
}