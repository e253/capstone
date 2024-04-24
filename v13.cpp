#include "common.hpp"
#include <cassert>
#include <cstdint>
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

#define INFLATE_WEIGHTS(w, w_i32, w_f32, zeros) \
    w = _mm_sub_epi8(w, zeros);                 \
    __m512i w_i32 = _mm512_cvtepi8_epi32(w);    \
    __m512 w_f32 = _mm512_cvtepi32_ps(w_i32);

void q4f32s_ukernel(
    uint8_t* w, // Weight, offset from the global pointer
    uint64_t w_cs, // Col stride for weights
    float* scales, // Weight scales, offset from the global pointer
    uint64_t scales_cs, // Col stride for scales
    uint8_t* zeros, // Weight zeros, offset from the global pointer
    uint64_t zeros_cs, // Col stride for zeros
    float* in, // input, offset from the global pointer
    float* out, // out, offset from the global pointer
    uint64_t cols)
{
    // and mask
    __m128i and_mask = _mm_set1_epi8(0x0F);

    // Initialize accumulators
    __m512 acc1 = _mm512_loadu_ps(out + 0);
    __m512 acc2 = _mm512_loadu_ps(out + 16);
    __m512 acc3 = _mm512_loadu_ps(out + 32);
    __m512 acc4 = _mm512_loadu_ps(out + 48);
    __m512 acc5 = _mm512_loadu_ps(out + 64);
    __m512 acc6 = _mm512_loadu_ps(out + 80);
    __m512 acc7 = _mm512_loadu_ps(out + 96);
    __m512 acc8 = _mm512_loadu_ps(out + 112);

    __m128i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
    __m128i zeros1, zeros2, zeros3, zeros4, zeros5, zeros6, zeros7, zeros8;
    __m512 scale1, scale2, scale3, scale4, scale5, scale6, scale7, scale8;

    __m512 input;
    __m512i weights;
    for (int col = 0; col < cols; col++) {
        if (col % QBLOCK_SIZE == 0) {
            // Zeros
            __m512i zeros_data = _mm512_loadu_epi8(zeros);
            zeros1 = _mm512_extracti32x4_epi32(zeros_data, 0);
            zeros2 = _mm_shuffle_epi32(zeros1, _MM_SHUFFLE(3, 2, 3, 2));
            zeros3 = _mm512_extracti32x4_epi32(zeros_data, 1);
            zeros4 = _mm_shuffle_epi32(zeros3, _MM_SHUFFLE(3, 2, 3, 2));
            zeros5 = _mm512_extracti32x4_epi32(zeros_data, 2);
            zeros6 = _mm_shuffle_epi32(zeros5, _MM_SHUFFLE(3, 2, 3, 2));
            zeros7 = _mm512_extracti32x4_epi32(zeros_data, 3);
            zeros8 = _mm_shuffle_epi32(zeros7, _MM_SHUFFLE(3, 2, 3, 2));
            UNZIP_U4_TO_U8(zeros1, tmp1, and_mask);
            UNZIP_U4_TO_U8(zeros2, tmp2, and_mask);
            UNZIP_U4_TO_U8(zeros3, tmp3, and_mask);
            UNZIP_U4_TO_U8(zeros4, tmp4, and_mask);
            UNZIP_U4_TO_U8(zeros5, tmp5, and_mask);
            UNZIP_U4_TO_U8(zeros6, tmp6, and_mask);
            UNZIP_U4_TO_U8(zeros7, tmp7, and_mask);
            UNZIP_U4_TO_U8(zeros8, tmp8, and_mask);
            zeros += zeros_cs;
        }

        // load input
        input = _mm512_set1_ps(in[col]);

        // load weights
        weights = _mm512_loadu_si512(w);
        __m128i weight1 = _mm512_extracti32x4_epi32(weights, 0); // get 32 4 bit values
        __m128i weight2 = _mm_shuffle_epi32(weight1, _MM_SHUFFLE(3, 2, 3, 2)); // copy upper 16 4 bit values to separate reg
        __m128i weight3 = _mm512_extracti32x4_epi32(weights, 1);
        __m128i weight4 = _mm_shuffle_epi32(weight3, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i weight5 = _mm512_extracti32x4_epi32(weights, 2);
        __m128i weight6 = _mm_shuffle_epi32(weight5, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i weight7 = _mm512_extracti32x4_epi32(weights, 3);
        __m128i weight8 = _mm_shuffle_epi32(weight7, _MM_SHUFFLE(3, 2, 3, 2));
        UNZIP_U4_TO_U8(weight1, tmp1, and_mask);
        UNZIP_U4_TO_U8(weight2, tmp2, and_mask);
        UNZIP_U4_TO_U8(weight3, tmp3, and_mask);
        UNZIP_U4_TO_U8(weight4, tmp4, and_mask);
        UNZIP_U4_TO_U8(weight5, tmp5, and_mask);
        UNZIP_U4_TO_U8(weight6, tmp6, and_mask);
        UNZIP_U4_TO_U8(weight7, tmp7, and_mask);
        UNZIP_U4_TO_U8(weight8, tmp8, and_mask);
        INFLATE_WEIGHTS(weight1, w1_i32, w1_f32, zeros1);
        INFLATE_WEIGHTS(weight2, w2_i32, w2_f32, zeros2);
        INFLATE_WEIGHTS(weight3, w3_i32, w3_f32, zeros3);
        INFLATE_WEIGHTS(weight4, w4_i32, w4_f32, zeros4);
        INFLATE_WEIGHTS(weight5, w5_i32, w5_f32, zeros5);
        INFLATE_WEIGHTS(weight6, w6_i32, w6_f32, zeros6);
        INFLATE_WEIGHTS(weight7, w7_i32, w7_f32, zeros7);
        INFLATE_WEIGHTS(weight8, w8_i32, w8_f32, zeros8);
        w += w_cs;

        // Multiply and accumulate
        acc1 = _mm512_fmadd_ps(w1_f32, input, acc1);
        acc2 = _mm512_fmadd_ps(w2_f32, input, acc2);
        acc3 = _mm512_fmadd_ps(w3_f32, input, acc3);
        acc4 = _mm512_fmadd_ps(w4_f32, input, acc4);
        acc5 = _mm512_fmadd_ps(w5_f32, input, acc5);
        acc6 = _mm512_fmadd_ps(w6_f32, input, acc6);
        acc7 = _mm512_fmadd_ps(w7_f32, input, acc7);
        acc8 = _mm512_fmadd_ps(w8_f32, input, acc8);

        if ((col + 1) % QBLOCK_SIZE == 0) {
            // Scales
            scale1 = _mm512_loadu_ps(scales + 0);
            scale2 = _mm512_loadu_ps(scales + 16);
            scale3 = _mm512_loadu_ps(scales + 32);
            scale4 = _mm512_loadu_ps(scales + 48);
            scale5 = _mm512_loadu_ps(scales + 64);
            scale6 = _mm512_loadu_ps(scales + 80);
            scale7 = _mm512_loadu_ps(scales + 96);
            scale8 = _mm512_loadu_ps(scales + 112);
            scales += scales_cs;

            // Scale
            acc1 = _mm512_mul_ps(acc1, scale1);
            acc2 = _mm512_mul_ps(acc2, scale2);
            acc3 = _mm512_mul_ps(acc3, scale3);
            acc4 = _mm512_mul_ps(acc4, scale4);
            acc5 = _mm512_mul_ps(acc5, scale5);
            acc6 = _mm512_mul_ps(acc6, scale6);
            acc7 = _mm512_mul_ps(acc7, scale7);
            acc8 = _mm512_mul_ps(acc8, scale8);
        }
    }

    // Store Results
    _mm512_storeu_ps(out, acc1);
    _mm512_storeu_ps(out + 16, acc2);
    _mm512_storeu_ps(out + 32, acc3);
    _mm512_storeu_ps(out + 48, acc4);
    _mm512_storeu_ps(out + 64, acc5);
    _mm512_storeu_ps(out + 80, acc6);
    _mm512_storeu_ps(out + 96, acc7);
    _mm512_storeu_ps(out + 112, acc8);
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

        const int n_col_blocks = n / 2048;

        for (int j = start_row; j < end_row; j += 128) {
            q4f32s_ukernel_prelude();
            for (int col_block = 0; col_block < n_col_blocks; col_block++) {
                int i = col_block * 2048;
                q4f32s_ukernel(
                    CM(w, j, i / 2, m / 2), m / 2,
                    CM(s, j, i / QBLOCK_SIZE, m), m,
                    CM(z, j, i / QBLOCK_SIZE / 2, m / 2), m / 2,
                    in, out + j,
                    2048);
            }
            q4f32s_ukernel_epiloque(out + j);
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
