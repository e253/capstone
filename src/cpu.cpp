#include "capstone/capstone.hpp"
#include "capstone/thread.hpp"
#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

void f32_qi8f32s(float* in, int8_t* out, float* out_s, int n, int n_threads)
{
    assert(n % QBLOCK_SIZE == 0 && "n must be a multiple of QBLOCK_SIZE");
    assert((n / n_threads) % QBLOCK_SIZE == 0 && "n_threads must not divide the qblock size below QBLOCK_SIZE");

    vector<pthread_t> threads(n_threads);
    vector<f32_qi8f32s_params> params(n_threads);

    for (int i = 0; i < n_threads - 1; i++) {
        params[i].in = in;
        params[i].out = out;
        params[i].out_s = out_s;
        params[i].n = n;
        params[i].tid = i;
        params[i].n_threads = n_threads;
        pthread_create(&threads[i], nullptr, f32_qi8f32s_thread, (void*)&params[i]);
    }

    struct f32_qi8f32s_params last_params = {
        in, out, out_s, n, n_threads - 1, n_threads
    };
    f32_qi8f32s_thread(&last_params);

    for (auto& t : threads) {
        pthread_join(t, nullptr);
    }

    threads.clear();
    params.clear();
}

thread_ret_t f32_qi8f32s_thread(void* _params)
{
    struct f32_qi8f32s_params* params = (f32_qi8f32s_params*)_params;

    float* in = params->in;
    int8_t* out = params->out;
    float* out_s = params->out_s;
    int n = params->n;
    int start = params->tid * (n / params->n_threads);
    int end = (params->tid + 1) * (n / params->n_threads);

    assert(n % QBLOCK_SIZE == 0 && "n must be a multiple of QBLOCK_SIZE");

    for (int j = start; j < end; j += QBLOCK_SIZE) { // qblock
        __m512 zero = _mm512_abs_ps(_mm512_loadu_ps(in + j));
        __m512 one = _mm512_abs_ps(_mm512_loadu_ps(in + j + 16));
        __m512 two = _mm512_abs_ps(_mm512_loadu_ps(in + j + 32));
        __m512 three = _mm512_abs_ps(_mm512_loadu_ps(in + j + 48));
        __m512 four = _mm512_abs_ps(_mm512_loadu_ps(in + j + 64));
        __m512 five = _mm512_abs_ps(_mm512_loadu_ps(in + j + 80));
        __m512 six = _mm512_abs_ps(_mm512_loadu_ps(in + j + 96));
        __m512 seven = _mm512_abs_ps(_mm512_loadu_ps(in + j + 112));

        __m512 max_value = _mm512_max_ps(zero, one);
        __m512 max_value2 = _mm512_max_ps(two, three);
        __m512 max_value3 = _mm512_max_ps(four, five);
        __m512 max_value4 = _mm512_max_ps(six, seven);
        max_value = _mm512_max_ps(max_value, max_value2);
        max_value3 = _mm512_max_ps(max_value3, max_value4);

        max_value = _mm512_max_ps(max_value, max_value3);

        float max_v = _mm512_reduce_max_ps(max_value);

        float scale = max_v > 127.0f ? (max_v / 127.0f) : 1.0f;

        out_s[j / QBLOCK_SIZE] = scale;

        for (int i = j; i < j + QBLOCK_SIZE; i++) { // values
            int8_t v = static_cast<int8_t>(round(in[i] / scale));
            out[i] = v;
        }
    }
    return 0;
}

/*
// gets 64 u4 from memory and expands to u8
static inline __m512i load_weights(const uint8_t* w)
{
    //    Rough asm implementation
    //    vmodqu8   (%rsi), %ymm0
    //    vmovdu8   %ymm0, %ymm1
    //    vpsrlw    $4, %ymm0, %ymm0
    //    vpunpcklo %zmm0, %ymm1, %zmm0
    //    vpand     %zmm0, %zmm2, %zmm0
    //    # zmm2 is set of 0x0F
    __m256i tmp = _mm256_loadu_si256((const __m256i*)w);
    __m512i weight = _mm512_unpacklo_epi8(_mm512_castsi256_si512(_mm256_srli_epi32(tmp, 4)), _mm512_castsi256_si512(tmp));
    return _mm512_and_si512(weight, _mm512_set1_epi8(0x0F));
}

static inline int hsum_i32_16(__m512i x)
{
    __m256i a = _mm256_add_epi32(_mm512_castsi512_si256(x), _mm512_extracti32x8_epi32(x, 1));
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

void q4f32s_qi8f32s_128x512_ukernel_offline(
    uint8_t* __restrict w,
    uint64_t w_rs,
    float* __restrict scales,
    uint8_t* __restrict zeros,
    int8_t* __restrict in,
    float* __restrict in_scales,
    int8_t* __restrict out,
    float out_scale)
{
    for (int row = 0; row < 128; row++) {
        float acc = 0;

        for (int qblock = 0; qblock < 4; qblock++) {
            // Initialize accumulator
            __m512i acci = _mm512_setzero_si512();

            // Load Zero
            uint8_t _zero = zeros[(qblock * QBLOCK_SIZE + row) / 2];
            _zero >>= (!(row & 1) << 2); // 4 if row is even, 0 otherwise
            _zero &= 0x0F;
            __m512i zero = _mm512_set1_epi8(_zero);

            {
                // load input 64 values
                __m512i input = _mm512_loadu_epi8(in + qblock * QBLOCK_SIZE);
                __m512i negative_input = _mm512_sub_epi8(_mm512_setzero_si512(), input);

                // load weights 64 values
                __m512i weight = load_weights(w + (qblock * QBLOCK_SIZE) / 2 + row * w_rs);

                // AVX512_VNNI likes u8 * i8 multiplications
                // input(i8) * (weights(u8) - zeros(u8)) == weights(u8)*input(i8) + zeros(u8)*(-input(i8))
                // Signed muliplications are possible, but only with 256-bit regs ^_^
                // - overflow safety
                // - highest weight value is 15, highest abs input is 128. at most 1920
                // - we can safely accumulate ~1M values onto i32, well within dimensions limits
                acci = _mm512_dpbusd_epi32(acci, weight, input);
                acci = _mm512_dpbusd_epi32(acci, zero, negative_input);
            }

            {
                __m512i input = _mm512_loadu_epi8(in + qblock * QBLOCK_SIZE + 64);
                __m512i negative_input = _mm512_sub_epi8(_mm512_setzero_si512(), input);
                __m512i weight = load_weights(w + (qblock * QBLOCK_SIZE + 64) / 2 + row * w_rs);
                acci = _mm512_dpbusd_epi32(acci, weight, input);
                acci = _mm512_dpbusd_epi32(acci, zero, negative_input);
            }

            acc += hsum_i32_16(acci) * scales[qblock * QBLOCK_SIZE + row] * in_scales[qblock];
        }

        out[row] += acc;
    }
}

void q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    int8_t* out,
    int m, int n)
{
    assert(m >= 512 && n >= 512 && "m and n must be at least 128");
    assert(m <= 32768 && n <= 32768 && "m and n can be at most 32768");
    assert(m % 512 == 0 && n % 512 == 0 && "m and n must be multiples of 128");

    size_t n_threads = 3;
    vector<thread> threads(n_threads);

    int rows_per_thread = m / (n_threads + 1);
    int start_row = 0;
    int end_row;

    for (int thread_id = 0; thread_id < n_threads; thread_id++) {
        end_row = start_row + rows_per_thread;
        threads[thread_id] = thread([&](uint8_t* w, float* s, uint8_t* z,
                                        int8_t* in, float* in_scales,
                                        int8_t* out, float* out_scales,
                                        int m, int n,
                                        int start_row, int end_row) {
            int in_qblocks = n / QBLOCK_SIZE;
            for (int col = 0; col < n; col += 512) {
                int in_qblock = col / QBLOCK_SIZE;
                for (int row = start_row; row < end_row; row += 128) {
                    int out_qblock = row / QBLOCK_SIZE;
                    int block_id = out_qblock * in_qblocks + in_qblock;

                    q4f32s_qi8f32s_128x512_ukernel_offline(
                        w + (row * n / 2 + col / 2), n / 2,
                        s + block_id * QBLOCK_SIZE,
                        z + block_id * (QBLOCK_SIZE / 2),
                        in + col, in_scales + col / QBLOCK_SIZE,
                        out + row, out_scales[row / QBLOCK_SIZE]);
                }
            }
        },
            w, s, z, in, in_scales, out, out_scales, m, n, start_row, end_row);

        start_row += rows_per_thread;
    }
    // this thread is a worker too!
    end_row += rows_per_thread;
    int in_qblocks = n / QBLOCK_SIZE;
    for (int col = 0; col < n; col += 512) {
        int in_qblock = col / QBLOCK_SIZE;
        for (int row = start_row; row < end_row; row += 128) {
            int out_qblock = row / QBLOCK_SIZE;
            int block_id = out_qblock * in_qblocks + in_qblock;

            q4f32s_qi8f32s_128x512_ukernel_offline(
                w + (row * n / 2 + col / 2), n / 2,
                s + block_id * QBLOCK_SIZE,
                z + block_id * (QBLOCK_SIZE / 2),
                in + col, in_scales + col / QBLOCK_SIZE,
                out + row, out_scales[row / QBLOCK_SIZE]);
        }
    }

    for (auto& t : threads) {
        t.join();
    }
    threads.clear();
}

// Testing!

#define BROADCAST(ptr, val, len)  \
    for (int i = 0; i < len; i++) \
    ptr[i] = val

void assert_arr_eq_i8(int8_t* arr, int8_t expected, int len, string passed_msg, string failed_msg)
{
    bool passed = true;
    for (int i = 0; i < len; i++) {
        if (arr[i] != expected) {
            cout << "Output[" << i << "] = " << (int)arr[i] << endl;
            passed = false;
        }
    }
    if (passed) {
        cout << passed_msg << endl;
    } else {
        cout << failed_msg << endl;
        exit(0);
    }
}

void test_128x512_offline()
{
    cout << "### q4f32s_qi8f32s_128x512_ukernel_offline() ###" << endl;

    int m = 128;
    int n = 512;

    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);
    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)_mm_malloc(n * sizeof(float), 64);
    int8_t* out = (int8_t*)_mm_malloc(m, 64);

    // 1 - Trivial
    {
        BROADCAST(w, 0x55, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        float in_s[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        memset(out, 0, m);
        float out_s = 81.96f;

        q4f32s_qi8f32s_128x512_ukernel_offline(
            w, n / 2,
            s, z,
            in, in_s,
            out, out_s);

        assert_arr_eq_i8(out, 100, m, "Offline 128x128 Test 1 Passed", "Offline 128x128 Test 1 Failed");
    }

    // 2 - non 1.0 input scale
    {
        BROADCAST(w, 0x55, m * n / 2);
        BROADCAST(s, 1.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);

        float in_s[] = { 2.0f, 2.0f, 2.0f, 2.0f };
        memset(out, 0, m);
        float out_s = 81.92f;

        q4f32s_qi8f32s_128x512_ukernel_offline(
            w, n / 2,
            s, z,
            in, in_s,
            out, out_s);

        assert_arr_eq_i8(out, 100, m, "Offline 128x128 Test 2 Passed", "Offline 128x128 Test 2 Failed");
    }

    // 3 - weights and scales alternated along the input dimension
    {
        for (int row = 0; row < m; row++) { // row idx
            for (int col = 0; col < n / 2; col++) { // col idx
                if (col % 2 == 0) {
                    w[row * n / 2 + col] = 0x33;
                } else {
                    w[row * n / 2 + col] = 0x55;
                }
            }
        }

        int n_col_qblocks = n / QBLOCK_SIZE;
        int n_row_qblocks = m / QBLOCK_SIZE;
        for (int row_block = 0; row_block < n_row_qblocks; row_block++) {
            for (int col_block = 0; col_block < n_col_qblocks; col_block++) {
                int block_id = row_block * n_row_qblocks + col_block;
                for (int el = 0; el < QBLOCK_SIZE; el++) {
                    s[block_id * QBLOCK_SIZE + el] = (col_block % 2 == 0) ? 1.0f : 2.0f;
                }
            }
        }

        BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        float in_s[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        memset(out, 0, m);
        float out_s = 46.08f;

        q4f32s_qi8f32s_128x512_ukernel_offline(
            w, n / 2,
            s, z,
            in, in_s,
            out, out_s);

        assert_arr_eq_i8(out, 100, m, "Offline 128x128 Test 3 Passed", "Offline 128x128 Test 3 Failed");
    }

    // 4 - Trivial - But with negative values after zero adjustment
    {
        memset(w, 0, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x44, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        float in_s[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        memset(out, 0, m);
        float out_s = 81.96f;

        q4f32s_qi8f32s_128x512_ukernel_offline(
            w, n / 2,
            s, z,
            in, in_s,
            out, out_s);

        assert_arr_eq_i8(out, -100, m, "Offline 128x128 Test 4 Passed", "Offline 128x128 Test 4 Failed");
    }

    // 5 - Trivial - But the out_s is too small to put the outputs into int8 range
    {
        memset(w, 0, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x44, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        float in_s[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        memset(out, 0, m);
        float out_s = 1.0f;

        q4f32s_qi8f32s_128x512_ukernel_offline(
            w, n / 2,
            s, z,
            in, in_s,
            out, out_s);

        assert_arr_eq_i8(out, -128, m, "Offline 128x128 Test 5 Passed", "Offline 128x128 Test 5 Failed");
    }

    // 6 - alternating weights along the output dimension
    {
        for (int row = 0; row < m; row++) { // row idx
            for (int col = 0; col < n / 2; col++) { // col idx
                w[row * n / 2 + col] = (row % 2 == 0) ? 0x33 : 0x55;
            }
        }
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        float in_s[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        memset(out, 0, m);
        float out_s = 81.96f;

        q4f32s_qi8f32s_128x512_ukernel_offline(
            w, n / 2,
            s, z,
            in, in_s,
            out, out_s);

        bool passed = true;
        for (int i = 0; i < m; i++) {
            if (out[i] != (i % 2 == 0 ? 50 : 100)) {
                std::cout << "Output[" << i << "] = " << (int)out[i] << std::endl;
                passed = false;
            }
        }

        if (passed) {
            cout << "Offline 128x128 Test 6 Passed" << endl;
        } else {
            cout << "Offline 128x128 Test 6 Failed" << endl;
        }
    }

    // 7 - alternate zero values
    {
        BROADCAST(w, 0x11, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++) {
            z[i] = (i % 2 == 0) ? 0x11 : 0x33;
        }
        BROADCAST(in, 2, n);
        float in_s[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        memset(out, 0, m);
        float out_s = 40.96f;

        q4f32s_qi8f32s_128x512_ukernel_offline(
            w, n / 2,
            s, z,
            in, in_s,
            out, out_s);

        bool passed = true;
        for (int i = 0; i < m; i += 4) {
            if (out[i] != 0 || out[i + 1] != 0 || out[i + 2] != -100 || out[i + 3] != -100) {
                cout << "Output[" << i << "] = " << (int)out[i] << endl;
                cout << "Output[" << i + 1 << "] = " << (int)out[i + 1] << endl;
                cout << "Output[" << i + 2 << "] = " << (int)out[i + 2] << endl;
                cout << "Output[" << i + 3 << "] = " << (int)out[i + 3] << endl;
                passed = false;
            }
        }

        if (passed) {
            std::cout << "Offline 128x128 Test 7 Passed" << std::endl;
        } else {
            std::cout << "Offline 128x128 Test 7 Failed" << std::endl;
        }
    }

    _mm_free(w);
    _mm_free(s);
    _mm_free(z);
    _mm_free(in);
    _mm_free(out);
    cout << endl;
}

void test_offline_egemv()
{
    cout << "### q4f32s_qi8f32s_egemv_offline() ###" << endl;

    int m = 512;
    int n = 512;

    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);
    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)_mm_malloc(n, 64);
    float* in_scales = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);
    int8_t* out = (int8_t*)_mm_malloc(m, 64);
    float* out_scales = (float*)_mm_malloc(m / QBLOCK_SIZE * sizeof(float), 64);

    // 1 - Trivial
    {
        BROADCAST(w, 0x55, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        BROADCAST(in_scales, 1.0f, n / QBLOCK_SIZE);
        memset(out, 0, m);
        BROADCAST(out_scales, 81.92f, m / QBLOCK_SIZE);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        assert_arr_eq_i8(out, 100, m, "Offline EGEMV Test 1 Passed", "Offline EGEMV Test 1 Failed");
    }

    // 2 - weight scales alternated along the input dimension
    {
        BROADCAST(w, 0x55, m * n / 2);
        for (int row_block = 0; row_block < m / QBLOCK_SIZE; row_block++) {
            for (int col_block = 0; col_block < n / QBLOCK_SIZE; col_block++) {
                int block_id = row_block * n / QBLOCK_SIZE + col_block;
                for (int el = 0; el < QBLOCK_SIZE; el++) {
                    s[block_id * QBLOCK_SIZE + el] = (col_block % 2 == 0) ? 1.0f : 2.0f;
                }
            }
        }
        BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        BROADCAST(in_scales, 1.0f, n / QBLOCK_SIZE);
        memset(out, 0, m);
        BROADCAST(out_scales, 61.44f, m / QBLOCK_SIZE);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        assert_arr_eq_i8(out, 100, m, "Offline EGEMV Test 2 Passed", "Offline Egemv Test 2 Failed");
    }

    // 3 - trivial, but non 1.0 input scale
    {
        BROADCAST(w, 0x55, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        BROADCAST(in_scales, 2.0f, n / QBLOCK_SIZE);
        memset(out, 0, m);
        BROADCAST(out_scales, 163.84f, m / QBLOCK_SIZE);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        assert_arr_eq_i8(out, 100, m, "Offline EGEMV Test 3 Passed", "Offline EGEMV Test 3 Failed");
    }

    // 4 - trivial, but with negative values after 0 adjustment
    {
        BROADCAST(w, 0x00, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x44, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        BROADCAST(in_scales, 1.0f, n / QBLOCK_SIZE);
        memset(out, 0, m);
        BROADCAST(out_scales, 81.92f, m / QBLOCK_SIZE);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        assert_arr_eq_i8(out, -100, m, "Offline EGEMV Test 4 Passed", "Offline EGEMV Test 4 Failed");
    }

    // 5 - trivial, but scale is too small to prevent int8 overflow
    {
        BROADCAST(w, 0x00, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x44, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        BROADCAST(in_scales, 1.0f, n / QBLOCK_SIZE);
        memset(out, 0, m);
        BROADCAST(out_scales, 1.0f, m / QBLOCK_SIZE);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        assert_arr_eq_i8(out, -128, m, "Offline EGEMV Test 5 Passed", "Offline EGEMV Test 5 Failed");
    }

    // 6 - alternating weights along the output dimension
    {
        for (int row = 0; row < m; row++) { // row idx
            for (int col = 0; col < n / 2; col++) { // col idx
                w[row * n / 2 + col] = (row % 2 == 0) ? 0x33 : 0x55;
            }
        }
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
        BROADCAST(in, 2, n);
        BROADCAST(in_scales, 1.0f, n / QBLOCK_SIZE);
        memset(out, 0, m);
        BROADCAST(out_scales, 81.92f, m / QBLOCK_SIZE);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        bool passed = true;
        for (int i = 0; i < m; i++) {
            if (out[i] != (i % 2 == 0 ? 50 : 100)) { // this is not 50, becuase of rounding.
                std::cout << "Output[" << i << "] = " << (int)out[i] << std::endl;
                passed = false;
            }
        }

        if (passed) {
            std::cout << "Offline EGEMV Test 6 Passed" << std::endl;
        } else {
            std::cout << "Offline EGEMV Test 6 Failed" << std::endl;
        }
    }

    // 7 - alternating zeros along out dim
    {
        BROADCAST(w, 0x11, m * n / 2);
        BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
        for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++) {
            z[i] = (i % 2 == 0) ? 0x11 : 0x33;
        }
        BROADCAST(in, 2, n);
        BROADCAST(in_scales, 1.0f, n / QBLOCK_SIZE);
        memset(out, 0, m);
        BROADCAST(out_scales, 40.96f, m / QBLOCK_SIZE);

        q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

        bool passed = true;
        for (int i = 0; i < m; i += 4) {
            if (out[i] != 0 || out[i + 1] != 0 || out[i + 2] != -100 || out[i + 3] != -100) {
                std::cout << "Output[" << i << "] = " << (int)out[i] << std::endl;
                std::cout << "Output[" << i + 1 << "] = " << (int)out[i + 1] << std::endl;
                std::cout << "Output[" << i + 2 << "] = " << (int)out[i + 2] << std::endl;
                std::cout << "Output[" << i + 3 << "] = " << (int)out[i + 3] << std::endl;
                passed = false;
            }
        }

        if (passed) {
            std::cout << "Offline EGEMV Test 7 Passed" << std::endl;
        } else {
            std::cout << "Offline EGEMV Test 7 Failed" << std::endl;
        }
    }

    _mm_free(w);
    _mm_free(s);
    _mm_free(z);
    _mm_free(in);
    _mm_free(in_scales);
    _mm_free(out);
    _mm_free(out_scales);
    cout << endl;
}

void test_dim_fuzz()
{
    cout << "### Dimension Fuzzing ###" << endl;

    uint8_t* w = (uint8_t*)_mm_malloc(32768 * 32768 / 2, 64);
    float* s = (float*)_mm_malloc(32768 * 32768 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)_mm_malloc(32768 * 32768 / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)_mm_malloc(32768, 64);
    float* in_scales = (float*)_mm_malloc(32768 / QBLOCK_SIZE * sizeof(float), 64);
    int8_t* out = (int8_t*)_mm_malloc(32768, 64);
    float* out_scales = (float*)_mm_malloc(32768 / QBLOCK_SIZE * sizeof(float), 64);

    for (int m = 512; m <= 32768; m += 512) {
        for (int n = 512; n <= 32768; n += 512) {

            // 1 - Trivial
            {
                BROADCAST(w, 0x55, m * n / 2);
                BROADCAST(s, 2.0f, m * n / QBLOCK_SIZE);
                BROADCAST(z, 0x11, m * n / QBLOCK_SIZE / 2);
                BROADCAST(in, 2, n);
                BROADCAST(in_scales, 1.0f, n / QBLOCK_SIZE);
                memset(out, 0, m);
                float _os = (float)(n * 2 * 2 * 4) / 100.0f;
                BROADCAST(out_scales, _os, m / QBLOCK_SIZE);

                q4f32s_qi8f32s_egemv_offline(w, s, z, in, in_scales, out, out_scales, m, n);

                int expected = round((float)8192 / _os) * (n / 512);
                if (expected > 127) {
                    expected = 127;
                }

                bool passed = true;
                for (int i = 0; i < m; i++) {
                    if (out[i] != expected) {
                        cout << "Output[" << i << "] = " << (int)out[i] << endl;
                        passed = false;
                    }
                }
                if (!passed) {
                    cout << "Fuzz Test 1 Failed for dim: " << m << ", " << n << endl;
                    cout << "Output scale: " << _os << endl;
                    cout << "Expected: " << expected << endl;
                    exit(0);
                }
            }
            cout << "Fuzz Test 1 Passed for dim: " << m << ", ?" << endl;
        }
    }

    _mm_free(w);
    _mm_free(s);
    _mm_free(z);
    _mm_free(in);
    _mm_free(in_scales);
    _mm_free(out);
    _mm_free(out_scales);
}

void random_init_array(char* arr, int len)
{
    for (int i = 0; i < len; i++) {
        arr[i] = rand() % 256;
    }
}

void bench_llama_up_proj()
{
    cout << "Benchmarking LLAMA Up Proj ..." << endl;
    cout << "Hidden Dim: 14336, Dim: 4096" << endl;
    cout << endl;

    int m = 14336;
    int n = 4096;

    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);
    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)_mm_malloc(n, 64);
    float* input_scales = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);
    int8_t* out = (int8_t*)_mm_malloc(m, 64);
    float* output_scales = (float*)_mm_malloc(m / QBLOCK_SIZE * sizeof(float), 64);
    random_init_array((char*)w, m * n / 2);
    random_init_array((char*)s, m * n / QBLOCK_SIZE * sizeof(float));
    random_init_array((char*)z, m * n / QBLOCK_SIZE / 2);
    random_init_array((char*)in, n);
    random_init_array((char*)input_scales, n / QBLOCK_SIZE * sizeof(float));

    const int NIT = 200;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        q4f32s_qi8f32s_egemv_offline(
            w, s, z,
            in, input_scales,
            out, output_scales,
            m, n);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = 4096 * 14336 * 2 * (uint64_t)NIT;
    double flops_per_sec = flops_processed / sec;
    cout << "GFLOPS: " << flops_per_sec / 1e9 << endl;
    cout << endl;

    _mm_free(w);
    _mm_free(s);
    _mm_free(z);
    _mm_free(in);
    _mm_free(input_scales);
    _mm_free(out);
    _mm_free(output_scales);
}

void bench_llama_ffn()
{
    // down proj 4096x14336
    // gate_proj 14336x4096
    // up_proj 14336x4096
    // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
    // we do not do the elementwise multiply
    // we do not apply SwiGLU non-linearity in the hidden_dim
    // just the matmuls. skips 14k ops out of 176M total (4096 * 14336 * 3)
    cout << "Benchmarking LLAMA FFN ..." << endl;
    cout << "down_proj @ (up_proj @ x * gate_proj @ x)" << endl;
    cout << "Hidden Dim: 14436, Dim: 4096" << endl;
    cout << endl;

    // ==== activations ====
    int8_t* io = (int8_t*)_mm_malloc(4096, 64);
    float* input_scales = (float*)_mm_malloc(4096 / QBLOCK_SIZE * sizeof(float), 64);
    float* output_scales = (float*)_mm_malloc(4096 / QBLOCK_SIZE * sizeof(float), 64);
    random_init_array((char*)io, 4096);
    random_init_array((char*)input_scales, 4096 / QBLOCK_SIZE);
    random_init_array((char*)output_scales, 4096 / QBLOCK_SIZE);

    // ==== up_proj ====
    uint8_t* w_up_proj = (uint8_t*)_mm_malloc(14336 * 4096 / 2, 64);
    float* s_up_proj = (float*)_mm_malloc(14336 * 4096 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_up_proj = (uint8_t*)_mm_malloc(14336 * 4096 / QBLOCK_SIZE / 2, 64);
    int8_t* out_up_proj = (int8_t*)_mm_malloc(14336, 64);
    float* out_scales_up_proj = (float*)_mm_malloc(14336 / QBLOCK_SIZE * sizeof(float), 64);
    random_init_array((char*)w_up_proj, 14336 * 4096 / 2);
    random_init_array((char*)s_up_proj, 14336 * 4096 / QBLOCK_SIZE);
    random_init_array((char*)z_up_proj, 14336 * 4096 / QBLOCK_SIZE / 2);
    random_init_array((char*)out_up_proj, 14336);
    random_init_array((char*)out_scales_up_proj, 14336 / QBLOCK_SIZE);

    // ==== gate_proj ====
    uint8_t* w_gate_proj = (uint8_t*)_mm_malloc(4096 * 14336 / 2, 64);
    float* s_gate_proj = (float*)_mm_malloc(4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_gate_proj = (uint8_t*)_mm_malloc(4096 * 14336 / QBLOCK_SIZE / 2, 64);
    int8_t* out_gate_proj = (int8_t*)_mm_malloc(14336, 64);
    float* out_scales_gate_proj = (float*)_mm_malloc(14336 / QBLOCK_SIZE * sizeof(float), 64);
    random_init_array((char*)w_gate_proj, 4096 * 14336 / 2);
    random_init_array((char*)s_gate_proj, 4096 * 14336 / QBLOCK_SIZE);
    random_init_array((char*)z_gate_proj, 4096 * 14336 / QBLOCK_SIZE / 2);
    random_init_array((char*)out_gate_proj, 14336);
    random_init_array((char*)out_scales_gate_proj, 14336 / QBLOCK_SIZE);

    // ==== down_proj ====
    uint8_t* w_down_proj = (uint8_t*)_mm_malloc(4096 * 14336 / 2, 64);
    float* s_down_proj = (float*)_mm_malloc(4096 * 14336 / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z_down_proj = (uint8_t*)_mm_malloc(4096 * 14336 / QBLOCK_SIZE / 2, 64);
    random_init_array((char*)w_down_proj, 4096 * 14336 / 2);
    random_init_array((char*)s_down_proj, 4096 * 14336 / QBLOCK_SIZE);
    random_init_array((char*)z_down_proj, 4096 * 14336 / QBLOCK_SIZE / 2);

    // ==== bench ====
    const int NIT = 200;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
        // up_proj @ x
        q4f32s_qi8f32s_egemv_offline(
            w_up_proj, s_up_proj, z_up_proj,
            io, input_scales,
            out_up_proj, out_scales_up_proj,
            14336, 4096);

        // gate_proj @ x
        q4f32s_qi8f32s_egemv_offline(
            w_gate_proj, s_gate_proj, z_gate_proj,
            io, input_scales,
            out_gate_proj, out_scales_gate_proj,
            14336, 4096);

        // down_proj @ up_proj_out
        q4f32s_qi8f32s_egemv_offline(
            w_down_proj, s_down_proj, z_down_proj,
            out_up_proj, out_scales_up_proj,
            io, output_scales,
            4096, 14336);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = 4096 * 14336 * 6 * (uint64_t)NIT;
    double flops_per_sec = flops_processed / sec;
    cout << "GFLOPS: " << flops_per_sec / 1e9 << endl;
    cout << endl;

    // ==== cleanup ====
    _mm_free(io);
    _mm_free(input_scales);
    _mm_free(w_up_proj);
    _mm_free(s_up_proj);
    _mm_free(z_up_proj);
    _mm_free(out_up_proj);
    _mm_free(out_scales_up_proj);
    _mm_free(w_gate_proj);
    _mm_free(s_gate_proj);
    _mm_free(z_gate_proj);
    _mm_free(out_gate_proj);
    _mm_free(out_scales_gate_proj);
    _mm_free(w_down_proj);
    _mm_free(s_down_proj);
    _mm_free(z_down_proj);
}

int main(int argc, char** argv)
{
    test_128x512_offline();
    test_offline_egemv();

    if (argc == 2) {
        if (string(argv[1]) == "fuzz") {
            cout << "Fuzzing tests across all supported input dimensions ..." << endl;
            cout << "This will take a long time" << endl;
            test_dim_fuzz();
        }
    } else {
        cout << "Skipping Fuzz. run `./cpu.exe fuzz` to fuzz" << endl;
    }
    cout << endl;

    bench_llama_up_proj();
    bench_llama_ffn();
}
*/