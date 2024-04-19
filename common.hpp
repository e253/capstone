#pragma once

#include <cstdint>

#define QBLOCK_SIZE 128
// column major indexing helper
#define CM(ptr, row, col, col_stride) ((ptr) + (col * (col_stride)) + (row))

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

void q4f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    float* in,
    float* out,
    int m, int n);

void* _alloc(std::size_t size, std::size_t alignment);
void _free(void* ptr);