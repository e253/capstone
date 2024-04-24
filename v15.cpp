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

#define UNZIP_U4_TO_U8(res, tmp, and_mask)                                \
    tmp = _mm_shuffle_epi32(AVXPS_TO_SSEI(res), _MM_SHUFFLE(3, 2, 0, 1)); \
    res = SSEI_TO_AVXPS(_mm_srli_si128(AVXPS_TO_SSEI(res), 4));           \
    res = SSEI_TO_AVXPS(_mm_and_si128(AVXPS_TO_SSEI(res), and_mask));     \
    tmp = _mm_and_si128(tmp, and_mask);                                   \
    res = SSEI_TO_AVXPS(_mm_unpacklo_epi8(AVXPS_TO_SSEI(res), tmp))

// 128i casted to 512
#define SSEI_TO_AVXPS(reg) _mm512_castps128_ps512(_mm_castsi128_ps(reg))
#define AVXPS_TO_SSEI(reg) _mm_castps_si128(_mm512_castps512_ps128(reg))

#define INFLATE_WEIGHTS(w, zeros)                                    \
    w = SSEI_TO_AVXPS(_mm_sub_epi8(AVXPS_TO_SSEI(w), zeros));        \
    w = _mm512_castsi512_ps(_mm512_cvtepi8_epi32(AVXPS_TO_SSEI(w))); \
    w = _mm512_cvtepi32_ps(_mm512_castps_si512(w));

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
    uint8_t* zeros,
    float* in,
    float* out)
{
// and mask
#define and_mask _mm_set1_epi8(0x0F)

    __m128i tmp; // compiler will create many regs for this
    for (int row = 0; row < 128; row += 4) {
        // Initialize accumulators
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();

        // Choose Zeros
        uint8_t zero12 = zeros[row / 2];
        uint8_t zero34 = zeros[row / 2 + 1];
        __m128i zero2 = _mm_set1_epi8(zero12 & 0x0F);
        __m128i zero1 = _mm_set1_epi8((zero12 >> 4) & 0x0F);
        __m128i zero4 = _mm_set1_epi8(zero34 & 0x0F);
        __m128i zero3 = _mm_set1_epi8((zero34 >> 4) & 0x0F);

        for (int col = 0; col < 128; col += 32) {
            // load input
            __m512 input = _mm512_loadu_ps(in + col);
            __m512 input2 = _mm512_loadu_ps(in + col + 16);

            // load weights
            __m512 weight1 = SSEI_TO_AVXPS(_mm_loadu_epi8(w + col / 2 + row * w_rs));
            __m512 weight2 = SSEI_TO_AVXPS(_mm_shuffle_epi32(AVXPS_TO_SSEI(weight1), _MM_SHUFFLE(3, 2, 3, 2)));
            __m512 weight3 = SSEI_TO_AVXPS(_mm_loadu_epi8(w + col / 2 + (row + 1) * w_rs));
            __m512 weight4 = SSEI_TO_AVXPS(_mm_shuffle_epi32(AVXPS_TO_SSEI(weight3), _MM_SHUFFLE(3, 2, 3, 2)));
            __m512 weight5 = SSEI_TO_AVXPS(_mm_loadu_epi8(w + col / 2 + (row + 2) * w_rs));
            __m512 weight6 = SSEI_TO_AVXPS(_mm_shuffle_epi32(AVXPS_TO_SSEI(weight5), _MM_SHUFFLE(3, 2, 3, 2)));
            __m512 weight7 = SSEI_TO_AVXPS(_mm_loadu_epi8(w + col / 2 + (row + 3) * w_rs));
            __m512 weight8 = SSEI_TO_AVXPS(_mm_shuffle_epi32(AVXPS_TO_SSEI(weight7), _MM_SHUFFLE(3, 2, 3, 2)));
            UNZIP_U4_TO_U8(weight1, tmp, and_mask);
            UNZIP_U4_TO_U8(weight2, tmp, and_mask);
            UNZIP_U4_TO_U8(weight3, tmp, and_mask);
            UNZIP_U4_TO_U8(weight4, tmp, and_mask);
            UNZIP_U4_TO_U8(weight5, tmp, and_mask);
            UNZIP_U4_TO_U8(weight6, tmp, and_mask);
            UNZIP_U4_TO_U8(weight7, tmp, and_mask);
            UNZIP_U4_TO_U8(weight8, tmp, and_mask);
            INFLATE_WEIGHTS(weight1, zero1);
            INFLATE_WEIGHTS(weight2, zero1);
            INFLATE_WEIGHTS(weight3, zero2);
            INFLATE_WEIGHTS(weight4, zero2);
            INFLATE_WEIGHTS(weight5, zero3);
            INFLATE_WEIGHTS(weight6, zero3);
            INFLATE_WEIGHTS(weight7, zero4);
            INFLATE_WEIGHTS(weight8, zero4);

            // Multiply and accumulate
            acc1 = _mm512_fmadd_ps(weight1, input, acc1);
            acc1 = _mm512_fmadd_ps(weight2, input2, acc1);
            acc2 = _mm512_fmadd_ps(weight3, input, acc2);
            acc2 = _mm512_fmadd_ps(weight4, input2, acc2);
            acc3 = _mm512_fmadd_ps(weight5, input, acc3);
            acc3 = _mm512_fmadd_ps(weight6, input2, acc3);
            acc4 = _mm512_fmadd_ps(weight7, input, acc4);
            acc4 = _mm512_fmadd_ps(weight8, input2, acc4);
        }

        // Scale
        __m512 scale1 = _mm512_set1_ps(scales[row]);
        __m512 scale2 = _mm512_set1_ps(scales[row + 1]);
        __m512 scale3 = _mm512_set1_ps(scales[row + 2]);
        __m512 scale4 = _mm512_set1_ps(scales[row + 3]);

        acc1 = _mm512_mul_ps(acc1, scale1);
        acc2 = _mm512_mul_ps(acc2, scale2);
        acc3 = _mm512_mul_ps(acc3, scale3);
        acc4 = _mm512_mul_ps(acc4, scale4);

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

        int n_row_blocks = m / QBLOCK_SIZE;
        for (int row = start_row; row < end_row; row += 128) {
            int row_block = row / QBLOCK_SIZE;
            for (int col = 0; col < n; col += 128) {
                int col_block = col / QBLOCK_SIZE;
                int block_id = row_block * n_row_blocks + col_block;

                q4f32s_128x128_ukernel(
                    w + (row * n / 2 + col / 2), n / 2,
                    s + block_id * QBLOCK_SIZE,
                    z + block_id * (QBLOCK_SIZE / 2),
                    in + col, out + row);
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
