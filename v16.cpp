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

// gets 64 u4 from memory and expands to u8
inline __m512i load_weights(const uint8_t* w)
{
    /* Rough asm implementation
    vmodqu8   (%rsi), %ymm0
    vmovdu8   %ymm0, %ymm1
    vpsrlw    $4, %ymm0, %ymm0
    vinserti32x8 $1, %zmm0, %ymm1, %zmm0
    vpand     %zmm0, %zmm2, %zmm0
    # zmm2 is set of 0x0F
    */
    __m256i tmp = _mm256_loadu_si256((const __m256i*)w);
    __m512i weight = _mm512_inserti32x8(_mm512_castsi256_si512(tmp), _mm256_srli_epi32(tmp, 4), 1);
    return _mm512_and_si512(weight, _mm512_set1_epi8(0x0F));
}

#ifdef I32ACCUM
typedef __m512i acc_t;
#define REDUCE_ADD(acc) _mm512_reduce_add_epi32((acc))
#else
typedef __m512 acc_t;
#define REDUCE_ADD(acc) _mm512_reduce_add_ps((acc))
#endif

inline void mul_input_weight_accum(__m512i input, __m512i weight, __m512i zero, acc_t acc)
{
    /*
    VNNI likes u8 * i8 multiplications
    input(i8) * (weights (u8) - zeros (u8)) == weights(u8)*input(i8) - zeros(u8)*input(i8)
    Signed muliplications are possible, but only with ymm registers, for some reason.
    */
    __m512i tmp1 = _mm512_dpbusds_epi32(_mm512_setzero_epi32(), weight, input);
    __m512i tmp2 = _mm512_dpbusds_epi32(_mm512_setzero_epi32(), zero, input);
    __m512i tmp1 = _mm512_sub_epi32(tmp1, tmp2);
#ifdef I32ACCUM
    acc = _mm512_add_epi32(acc, tmp1);
#else
    acc = _mm512_add_ps(acc, _mm512_cvtepi32_ps(tmp1));
#endif
}

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
void q4f32s_qi8f32s_128x128_ukernel(
    uint8_t* __restrict w,
    uint64_t w_rs,
    float* __restrict scales,
    uint8_t* __restrict zeros,
    uint8_t* __restrict in,
    float* __restrict in_scales,
    float* __restrict out,
    uint64_t cols)
{
    for (int row = 0; row < 128; row += 2) {
// Initialize accumulators
#ifdef I32ACCUM
        __m512i acc1 = _mm512_setzero_epi8();
        __m512i acc2 = _mm512_setzero_epi8();
#else
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
#endif

        // Choose Zeros
        uint8_t zero12 = zeros[row / 2];
        __m512i zero2 = _mm512_set1_epi8(zero12 & 0x0F);
        __m512i zero1 = _mm512_set1_epi8((zero12 >> 4) & 0x0F);

        for (int col = 0; col < 128; col += 64) {
            // load input 64 values
            __m512i input = _mm512_loadu_epi8(in + col);

            // load weights 64 values each.
            __m512i weight1 = load_weights(w + col / 2 + row * w_rs);
            __m512i weight2 = load_weights(w + col / 2 + (row + 1) * w_rs);

            mul_input_weight_accum(input, weight1, zero1, acc1);
            mul_input_weight_accum(input, weight2, zero2, acc2);
        }

        // Store Results
        out[row] += REDUCE_ADD(acc1) * scales[row] * in_scales[row];
        out[row + 1] += REDUCE_ADD(acc2) * scales[row + 1] * in_scales[row + 1];
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
