#include "common.hpp"
#include <immintrin.h>
#include <cassert>
#include <cstdint>
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

inline void vpsrld4(__m128i x) {
    asm volatile(
        "vpsrld $4, %[input], %[output]" 
        :[output]"+x"(x)
        :[input]"x"(x)
        :
    );
}

inline void cpy(__m128i& dst, __m128i src) {
    asm volatile(
        "vmovdqa64 %[src], %[dst]"
        :[dst]"+x"(dst)
        :[src]"x"(src)
        :
    );
}


// tmp = _mm_shuffle_epi32(res, _MM_SHUFFLE(0,0,2,2));
#define UNZIP_U4_TO_U8(ptr, res, tmp, and_mask)    \
    res = _mm_mask_loadu_epi8(res, 0x00FF, ptr);   \
    cpy(tmp, res);                                 \
    vpsrld4(res);                                  \
    res = _mm_and_si128(res, and_mask);            \
    tmp = _mm_and_si128(tmp, and_mask);            \
    res = _mm_unpacklo_epi8(res, tmp)

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
// 8 * 16 element blocks of columns. We load the inputs once
//        and_mask = xmm0
//        zeros = xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8
//        scales = zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15, zmm16
//        accs = zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24
//        weights = zmm25, zmm26, zmm27, zmm28
//        tmp = xmm30, xmm31
//        input = zmm29
//        Col 1: (block 0)
//           Tmp: xmm30
//           Weights: zmm25
//        Col 2: (block 1)
//           Tmp: xmm31
//           Weights: zmm26
//        Col 3: (block 2)
//            Tmp: xmm30
//            Weights: zmm27
//        Col 4: (block 3)
//            Tmp: xmm31
//            Weights: zmm28
//        Col 5: (block 0)
//            Tmp: xmm30
//            Weights: zmm25
//        Col 6: (block 1)
//            Tmp: xmm31
//            Weights: zmm26
//        Col 7: (block 2)
//            Tmp: xmm30
//            Weights: zmm27
//        Col 8: (block 3)
//            Tmp: xmm31
//            Weights: zmm28

    // and mask
    __m128i xmm0 = _mm_set1_epi8(0x0F);
    // input
    __m512 zmm29;
    // zeros
    __m128i xmm1;
    __m128i xmm2;
    __m128i xmm3;
    __m128i xmm4;
    __m128i xmm5;
    __m128i xmm6;
    __m128i xmm7;
    __m128i xmm8;
    // scales
    __m512 zmm9;
    __m512 zmm10;
    __m512 zmm11;
    __m512 zmm12;
    __m512 zmm13;
    __m512 zmm14;
    __m512 zmm15;
    __m512 zmm16;
    // accs
    __m512 zmm17;
    __m512 zmm18;
    __m512 zmm19;
    __m512 zmm20;
    __m512 zmm21;
    __m512 zmm22;
    __m512 zmm23;
    __m512 zmm24;
    // weights
    __m512 zmm25;
    __m512i zmm25i;
    __m512 zmm26;
    __m512i zmm26i;
    __m512 zmm27;
    __m512i zmm27i;
    __m512 zmm28;
    __m512i zmm28i;
    __m128i xmm25;
    __m128i xmm26;
    __m128i xmm27;
    __m128i xmm28;
    // tmp
    __m128i xmm30;
    __m128i xmm31;


    // Initialize zeros
    UNZIP_U4_TO_U8(zeros,    xmm1, xmm30, xmm0);
    UNZIP_U4_TO_U8(zeros+8,  xmm2, xmm31, xmm0);
    UNZIP_U4_TO_U8(zeros+16, xmm3, xmm30, xmm0);
    UNZIP_U4_TO_U8(zeros+24, xmm4, xmm31, xmm0);
    UNZIP_U4_TO_U8(zeros+32, xmm5, xmm30, xmm0);
    UNZIP_U4_TO_U8(zeros+40, xmm6, xmm31, xmm0);
    UNZIP_U4_TO_U8(zeros+48, xmm7, xmm30, xmm0);
    UNZIP_U4_TO_U8(zeros+56, xmm8, xmm31, xmm0);
    zeros += zeros_cs;

    // Initialize scales
    zmm9  = _mm512_loadu_ps(scales +   0);
    zmm10 = _mm512_loadu_ps(scales +  16);
    zmm11 = _mm512_loadu_ps(scales +  32);
    zmm12 = _mm512_loadu_ps(scales +  48);
    zmm13 = _mm512_loadu_ps(scales +  64);
    zmm14 = _mm512_loadu_ps(scales +  80);
    zmm15 = _mm512_loadu_ps(scales +  96);
    zmm16 = _mm512_loadu_ps(scales + 112);
    scales += scales_cs;

    // Initialize accumulators 
    zmm17 = _mm512_loadu_ps(out +   0);
    zmm18 = _mm512_loadu_ps(out +  16);
    zmm19 = _mm512_loadu_ps(out +  32);
    zmm20 = _mm512_loadu_ps(out +  48);
    zmm21 = _mm512_loadu_ps(out +  64);
    zmm22 = _mm512_loadu_ps(out +  80);
    zmm23 = _mm512_loadu_ps(out +  96);
    zmm24 = _mm512_loadu_ps(out + 112);

    for (int col = 0; col < cols; col++) {
        zmm29 = _mm512_set1_ps(in[col]);

        // ========== 1 ==========
        // Fetch Weights (u4) --> xmm2 (i8)
        UNZIP_U4_TO_U8(w + 0, xmm25, xmm30, xmm0);
        // Zero adjust weights (i8)
        xmm25 = _mm_sub_epi8(xmm25, xmm1);
        // Weights (i8) --> Weights (i32)
        zmm25i = _mm512_cvtepi8_epi32(xmm25);
        // Weight (i32) --> Weight (f32)
        zmm25 = _mm512_cvtepi32_ps(zmm25i);
        // Weight *= scale
        zmm25 = _mm512_mul_ps(zmm9, zmm25);
        // Acc += in * weights (f32)
        zmm17 = _mm512_fmadd_ps(zmm25, zmm29, zmm17);

        // ========== 2 ==========
        UNZIP_U4_TO_U8(w + 8, xmm26, xmm31, xmm0);
        xmm26 = _mm_sub_epi8(xmm26, xmm2);
        zmm26i = _mm512_cvtepi8_epi32(xmm26);
        zmm26 = _mm512_cvtepi32_ps(zmm26i);
        zmm26 = _mm512_mul_ps(zmm10, zmm26);
        zmm18 = _mm512_fmadd_ps(zmm26, zmm29, zmm18);

        // ========== 3 ==========
        UNZIP_U4_TO_U8(w + 16, xmm27, xmm30, xmm0);
        xmm27 = _mm_sub_epi8(xmm27, xmm3);
        zmm27i = _mm512_cvtepi8_epi32(xmm27);
        zmm27 = _mm512_cvtepi32_ps(zmm27i);
        zmm27 = _mm512_mul_ps(zmm11, zmm27);
        zmm19 = _mm512_fmadd_ps(zmm27, zmm29, zmm19);

        // ========== 4 ==========
        UNZIP_U4_TO_U8(w + 24, xmm28, xmm31, xmm0);
        xmm28 = _mm_sub_epi8(xmm28, xmm4);
        zmm28i = _mm512_cvtepi8_epi32(xmm28);
        zmm28 = _mm512_cvtepi32_ps(zmm28i);
        zmm28 = _mm512_mul_ps(zmm12, zmm28);
        zmm20 = _mm512_fmadd_ps(zmm28, zmm29, zmm20);

        // ========== 5 ==========
        UNZIP_U4_TO_U8(w + 32, xmm25, xmm30, xmm0);
        xmm25 = _mm_sub_epi8(xmm25, xmm5);
        zmm25i = _mm512_cvtepi8_epi32(xmm25);
        zmm25 = _mm512_cvtepi32_ps(zmm25i);
        zmm25 = _mm512_mul_ps(zmm13, zmm25);
        zmm21 = _mm512_fmadd_ps(zmm25, zmm29, zmm21);

        // ========== 6 ==========
        UNZIP_U4_TO_U8(w + 40, xmm26, xmm31, xmm0);
        xmm26 = _mm_sub_epi8(xmm26, xmm6);
        zmm26i = _mm512_cvtepi8_epi32(xmm26);
        zmm26 = _mm512_cvtepi32_ps(zmm26i);
        zmm26 = _mm512_mul_ps(zmm14, zmm26);
        zmm22 = _mm512_fmadd_ps(zmm26, zmm29, zmm22);

        // ========== 7 ==========
        UNZIP_U4_TO_U8(w + 48, xmm27, xmm30, xmm0);
        xmm27 = _mm_sub_epi8(xmm27, xmm7);
        zmm27i = _mm512_cvtepi8_epi32(xmm27);
        zmm27 = _mm512_cvtepi32_ps(zmm27i);
        zmm27 = _mm512_mul_ps(zmm15, zmm27);
        zmm23 = _mm512_fmadd_ps(zmm27, zmm29, zmm23);

        // ========== 8 ==========
        UNZIP_U4_TO_U8(w + 56, xmm28, xmm31, xmm0);
        xmm28 = _mm_sub_epi8(xmm28, xmm8);
        zmm28i = _mm512_cvtepi8_epi32(xmm28);
        zmm28 = _mm512_cvtepi32_ps(zmm28i);
        zmm28 = _mm512_mul_ps(zmm16, zmm28);
        zmm24 = _mm512_fmadd_ps(zmm28, zmm29, zmm24);

        w += w_cs;

        if (((col & QBLOCK_SIZE) == 0) && (col != 0)) {
            // Zeros
            UNZIP_U4_TO_U8(zeros,      xmm1, xmm30, xmm0);
            UNZIP_U4_TO_U8(zeros +  8, xmm2, xmm31, xmm0);
            UNZIP_U4_TO_U8(zeros + 16, xmm3, xmm30, xmm0);
            UNZIP_U4_TO_U8(zeros + 24, xmm4, xmm31, xmm0);
            UNZIP_U4_TO_U8(zeros + 32, xmm5, xmm30, xmm0);
            UNZIP_U4_TO_U8(zeros + 40, xmm6, xmm31, xmm0);
            UNZIP_U4_TO_U8(zeros + 48, xmm7, xmm30, xmm0);
            UNZIP_U4_TO_U8(zeros + 56, xmm8, xmm31, xmm0);
            zeros += zeros_cs;

            // Scales
            zmm9 =  _mm512_loadu_ps(scales +   0);
            zmm10 = _mm512_loadu_ps(scales +  16);
            zmm11 = _mm512_loadu_ps(scales +  32);
            zmm12 = _mm512_loadu_ps(scales +  48);
            zmm13 = _mm512_loadu_ps(scales +  64);
            zmm14 = _mm512_loadu_ps(scales +  80);
            zmm15 = _mm512_loadu_ps(scales +  96);
            zmm16 = _mm512_loadu_ps(scales + 112);
            scales += scales_cs;
        }
    }

    // Store Results
    _mm512_storeu_ps(out,      zmm17);
    _mm512_storeu_ps(out + 16, zmm18);
    _mm512_storeu_ps(out + 32, zmm19);
    _mm512_storeu_ps(out + 48, zmm20);
    _mm512_storeu_ps(out + 64, zmm21);
    _mm512_storeu_ps(out + 80, zmm22);
    _mm512_storeu_ps(out + 96, zmm23);
    _mm512_storeu_ps(out + 112, zmm24);
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
