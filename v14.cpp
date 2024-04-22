#include "common.hpp"
#include <cassert>
#include <immintrin.h>
#include <iostream>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

using namespace std;

// Set accumulators to zero (zmm9-zmm16)
void q4f32s_ukernel_prelude()
{
}

// Dump Accumulators to memory (zmm9-zmm16)
void q4f32s_ukernel_epiloque(float* ptr)
{
}

#define UNZIP_U4_TO_U8(res, tmp, and_mask)                 \
    tmp = _mm_shuffle_epi32(res, _MM_SHUFFLE(3, 2, 0, 1)); \
    res = _mm_srli_si128(res, 4);                          \
    res = _mm_and_si128(res, and_mask);                    \
    tmp = _mm_and_si128(tmp, and_mask);                    \
    res = _mm_unpacklo_epi8(res, tmp)

#define INFLATE_WEIGHTS(w, w_i32, w_f32, zeros, scales) \
    w = _mm_sub_epi8(w, zeros);                         \
    __m512i w_i32 = _mm512_cvtepi8_epi32(w);            \
    __m512 w_f32 = _mm512_cvtepi32_ps(w_i32);           \
    w_f32 = _mm512_mul_ps(scales, w_f32)

/*
w, Weight, offset from the global pointer
w_rs, Row stride for weights
scales, Weight scales, offset from the global pointer
scales_rs, Row stride for scales
zeros, Weight zeros, offset from the global pointer
zeros_cs, Col stride for zeros
in, input, offset from the global pointer
out, out, offset from the global pointer
*/
void q4f32s_128x128_ukernel(
    uint8_t* w,
    uint64_t w_rs,
    float* scales,
    uint64_t scales_cs,
    uint8_t* zeros,
    uint64_t zeros_cs,
    float* in,
    float* out)
{
    // and mask
    __m128i and_mask = _mm_set1_epi8(0x0F);

    __m128i tmp; // compiler will create many regs for this
    for (int row = 0; row < 128; row += 4) {
        // Initialize accumulators
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();

        // Choose Zeros --> TODO! Fix. Incorrect
        uint8_t zero12 = zeros[row / 2];
        uint8_t zero34 = zeros[row / 2 + 1];
        __m128i zero2 = _mm_set1_epi8(zero12 & 0x0F);
        __m128i zero1 = _mm_set1_epi8((zero12 >> 4) & 0x0F);
        __m128i zero4 = _mm_set1_epi8(zero34 & 0x0F);
        __m128i zero3 = _mm_set1_epi8((zero34 >> 4) & 0x0F);

        // Choose Scales
        __m512 scale1 = _mm512_set1_ps(scales[row]);
        __m512 scale2 = _mm512_set1_ps(scales[row + 1]);
        __m512 scale3 = _mm512_set1_ps(scales[row + 2]);
        __m512 scale4 = _mm512_set1_ps(scales[row + 3]);

        for (int col = 0; col < 128; col += 32) {
            // load input
            __m512 input = _mm512_loadu_ps(in + col);
            __m512 input2 = _mm512_loadu_ps(in + col + 16);

            // load weights
            __m128i weight1 = _mm_loadu_epi8(w + col / 2 + row * w_rs);
            __m128i weight2 = _mm_shuffle_epi32(weight1, _MM_SHUFFLE(3, 2, 3, 2));
            __m128i weight3 = _mm_loadu_epi8(w + col / 2 + (row + 1) * w_rs);
            __m128i weight4 = _mm_shuffle_epi32(weight3, _MM_SHUFFLE(3, 2, 3, 2));
            __m128i weight5 = _mm_loadu_epi8(w + col / 2 + (row + 2) * w_rs);
            __m128i weight6 = _mm_shuffle_epi32(weight5, _MM_SHUFFLE(3, 2, 3, 2));
            __m128i weight7 = _mm_loadu_epi8(w + col / 2 + (row + 3) * w_rs);
            __m128i weight8 = _mm_shuffle_epi32(weight7, _MM_SHUFFLE(3, 2, 3, 2));
            UNZIP_U4_TO_U8(weight1, tmp, and_mask);
            UNZIP_U4_TO_U8(weight2, tmp, and_mask);
            UNZIP_U4_TO_U8(weight3, tmp, and_mask);
            UNZIP_U4_TO_U8(weight4, tmp, and_mask);
            UNZIP_U4_TO_U8(weight5, tmp, and_mask);
            UNZIP_U4_TO_U8(weight6, tmp, and_mask);
            UNZIP_U4_TO_U8(weight7, tmp, and_mask);
            UNZIP_U4_TO_U8(weight8, tmp, and_mask);
            INFLATE_WEIGHTS(weight1, w1_i32, w1_f32, zero1, scale1);
            INFLATE_WEIGHTS(weight2, w2_i32, w2_f32, zero1, scale1);
            INFLATE_WEIGHTS(weight3, w3_i32, w3_f32, zero2, scale2);
            INFLATE_WEIGHTS(weight4, w4_i32, w4_f32, zero2, scale2);
            INFLATE_WEIGHTS(weight5, w5_i32, w5_f32, zero3, scale3);
            INFLATE_WEIGHTS(weight6, w6_i32, w6_f32, zero3, scale3);
            INFLATE_WEIGHTS(weight7, w7_i32, w7_f32, zero4, scale4);
            INFLATE_WEIGHTS(weight8, w8_i32, w8_f32, zero4, scale4);

            // Multiply and accumulate
            acc1 = _mm512_fmadd_ps(w1_f32, input, acc1);
            acc1 = _mm512_fmadd_ps(w2_f32, input2, acc1);
            acc2 = _mm512_fmadd_ps(w3_f32, input, acc2);
            acc2 = _mm512_fmadd_ps(w4_f32, input2, acc2);
            acc3 = _mm512_fmadd_ps(w5_f32, input, acc3);
            acc3 = _mm512_fmadd_ps(w6_f32, input2, acc3);
            acc4 = _mm512_fmadd_ps(w7_f32, input, acc4);
            acc4 = _mm512_fmadd_ps(w8_f32, input2, acc4);
        }

        // Store Results
        out[row] += _mm512_reduce_add_ps(acc1);
        out[row + 1] += _mm512_reduce_add_ps(acc2);
        out[row + 2] += _mm512_reduce_add_ps(acc3);
        out[row + 3] += _mm512_reduce_add_ps(acc4);
    }
}

void q4f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    float* in,
    float* out,
    int m, int n)
{
    assert(m % 128 == 0 && "Row size must be divisble by 128");
    assert(n % QBLOCK_SIZE == 0 && "Col size must be divisble by 128");

    auto process_128_rows_n_cols = [&](uint8_t* w, float* s, uint8_t* z,
                                       float* in, float* out,
                                       int m, int n,
                                       int start_row, int end_row,
                                       int thread_id) {
#ifdef _WIN32
        // Doesn't seem to be helpful
        HANDLE hThread = GetCurrentThread();
        DWORD_PTR mask = 1 << (thread_id * 2);
        DWORD_PTR result = SetThreadAffinityMask(hThread, mask);
        if (result == 0) {
            cerr << "Error calling SetThreadAffinityMask: " << GetLastError() << endl;
            exit(0);
        }
#endif

        for (int row = start_row; row < end_row; row += 128) {
            for (int col = 0; col < n; col += 128) {
                q4f32s_128x128_ukernel(
                    w + (row * n / 2 + col), n / 2,
                    CM(s, row, col / QBLOCK_SIZE, m), m,
                    CM(z, row, col / QBLOCK_SIZE / 2, m / 2), m / 2,
                    in, out + row);
            }
        }
    };

    size_t n_threads = 4;
    vector<thread> threads(n_threads);
#ifndef _WIN32
    cpu_set_t cpuset;
#endif

    int rows_per_thread = m / n_threads;
    assert(rows_per_thread % 128 == 0 && "Thread row blocks size must be divisible by 128");
    int start_row = 0;
    int end_row;

    for (int thread_id = 0; thread_id < n_threads; thread_id++) {
        end_row = start_row + rows_per_thread;
        threads[thread_id] = thread(process_128_rows_n_cols, w, s, z, in, out, m, n, start_row, end_row, thread_id);
#ifndef _WIN32
        // https://eli.thegreenplace.net/2016/c11-threads-affinity-and-hyperthreading/
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        int rc = pthread_setaffinity_np(threads[thread_id].native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            cerr << "Error calling pthread_setaffinity_np: " << rc << endl;
            exit(0);
        }
#endif
        start_row += rows_per_thread;
    }
    for (auto& t : threads) {
        t.join();
    }
    threads.clear();
}
