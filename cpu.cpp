#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

// gets 64 u4 from memory and expands to u8
inline __m512i load_weights(const uint8_t* w)
{
    /*
        Rough asm implementation
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

typedef __m512i acc_t;
#define CLAMP(x, lo, hi) (x < lo ? lo : (x > hi ? hi : round(x)))
#define QBLOCK_SIZE 128

inline acc_t mul_input_weight_accum(__m512i input, __m512i negative_input, __m512i weight, __m512i zero, acc_t acc)
{
    /*
        AVX512_VNNI likes u8 * i8 multiplications
        input(i8) * (weights(u8) - zeros(u8)) == weights(u8)*input(i8) - zeros(u8)*input(i8)
        Signed muliplications are possible, but only with 256-bit regs ^_^

        - overflow safety
        - highest weight value is 15, highest abs input is 128. at most 1920
        - highest zero value 15, highest abs input is 128. at most 1920
        - most this change the accumulator is 11360 (4 values are added into each i32 in acc) * (3480 for each weight-zero pair)
        - we can safely use i32 for accumulation for ~600k values
        - Or if we want to reduce these 16 i32 values to one, we use ~35k values
        - This is within the dimension limits [512 - 32768]
    */

    acc = _mm512_dpbusds_epi32(acc, weight, input);
    acc = _mm512_dpbusds_epi32(acc, zero, negative_input);
    return acc;
}

// https://github.com/ggerganov/ggml/blob/cf1acc512432a969cee18947330fd26df67fb456/src/ggml-quants.c#L84
inline float hsum_float_16(__m512 x)
{
    __m256 tmp256 = _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
    __m128 res = _mm256_extractf128_ps(tmp256, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(tmp256));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

/*
w, Weight, offset from the global pointer
w_rs, Row stride for weights
scales, Weight scales, offset from the global pointer
zeros, Weight zeros, offset from the global pointer
in, input, offset from the global pointer
in_scale, scale value for input
out, out, offset from the global pointer
out_scale, scale value for output
*/
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
        __m512 acc = _mm512_setzero_ps();

        for (int qblock = 0; qblock < 4; qblock++) {
            // Initialize accumulators
            __m512i acci = _mm512_setzero_si512();

            // Load Zero
            uint8_t _zero = zeros[(qblock * QBLOCK_SIZE + row) / 2];
            _zero >>= (!(row & 1) << 2); // 4 if row is even, 0
            _zero &= 0x0F;
            __m512i zero = _mm512_set1_epi8(_zero);

            for (int col = 0; col < 128; col += 64) {
                // load input 64 values
                __m512i input = _mm512_loadu_epi8(in + qblock * QBLOCK_SIZE + col);
                __m512i negative_input = _mm512_sub_epi8(_mm512_setzero_si512(), input);

                // load weights 64 values each
                __m512i weight = load_weights(w + (qblock * QBLOCK_SIZE + col) / 2 + row * w_rs);

                acci = _mm512_dpbusds_epi32(acci, weight, input);
                acci = _mm512_dpbusds_epi32(acci, zero, negative_input);
            }

            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acci), _mm512_set1_ps(scales[qblock * QBLOCK_SIZE + row] * in_scales[qblock]), acc);
        }

        out[row] = (int8_t)CLAMP((out[row] + hsum_float_16(acc) / out_scale), -128.0f, 127.0f);
    }
}

void q4f32s_qi8f32s_egemv_offline(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    int8_t* out,
    float* out_scales,
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
/*
w, Weight, offset from the global pointer
w_rs, Row stride for weights
scales, Weight scales, offset from the global pointer
scales_rs, Row stride for scales
zeros, Weight zeros, offset from the global pointer
zeros_cs, Col stride for zeros
in, input, offset from the global pointer
out, out, offset from the global pointer
void q4f32s_qi8f32s_128x128_ukernel_otf(
    uint8_t* __restrict w,
    uint64_t w_rs,
    float* __restrict scales,
    uint8_t* __restrict zeros,
    int8_t* __restrict in,
    float in_scale,
    float* __restrict out)
{

    for (int row = 0; row < 128; row += 4) {

// Initialize accumulators
#ifdef I32ACCUM
        __m512i acc1 = _mm512_setzero_epi32();
        __m512i acc2 = _mm512_setzero_epi32();
        __m512i acc3 = _mm512_setzero_epi32();
        __m512i acc4 = _mm512_setzero_epi32();
#else
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();
#endif

        // Choose Zeros
        uint8_t _zero1 = zeros[row / 2];
        uint8_t _zero2 = _zero1; // copy?
        uint8_t _zero3 = zeros[row / 2 + 1];
        uint8_t _zero4 = _zero3;
        __m512i zero1 = _mm512_set1_epi8((_zero1 >> 4) & 0x0F);
        __m512i zero2 = _mm512_set1_epi8(_zero2 & 0x0F);
        __m512i zero3 = _mm512_set1_epi8((_zero3 >> 4) & 0x0F);
        __m512i zero4 = _mm512_set1_epi8(_zero4 & 0x0F);

        for (int col = 0; col < 128; col += 64) {
            // load input 64 values
            __m512i input = _mm512_loadu_epi8(in + col);

            // load weights 64 values each.
            __m512i weight1 = load_weights(w + col / 2 + row * w_rs);
            __m512i weight2 = load_weights(w + col / 2 + (row + 1) * w_rs);
            __m512i weight3 = load_weights(w + col / 2 + (row + 2) * w_rs);
            __m512i weight4 = load_weights(w + col / 2 + (row + 3) * w_rs);

            acc1 = mul_input_weight_accum(input, weight1, zero1, acc1);
            acc2 = mul_input_weight_accum(input, weight2, zero2, acc2);
            acc3 = mul_input_weight_accum(input, weight3, zero3, acc3);
            acc4 = mul_input_weight_accum(input, weight4, zero4, acc4);
        }

        // This could be more efficient
        out[row] += REDUCE_ADD(acc1) * scales[row] * in_scale;
        out[row + 1] += REDUCE_ADD(acc2) * scales[row + 1] * in_scale;
        out[row + 2] += REDUCE_ADD(acc3) * scales[row + 2] * in_scale;
        out[row + 3] += REDUCE_ADD(acc4) * scales[row + 3] * in_scale;
    }
}

void q4f32s_qi8f32s_egemv_otf(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    float* out,
    int m, int n)
{
    assert(m % 128 == 0 && "Row size must be divisble by 128");
    assert(n % QBLOCK_SIZE == 0 && "Col size must be divisble by 128");

    auto worker = [&](uint8_t* w, float* s, uint8_t* z,
                      int8_t* in, float* in_scales,
                      float* out,
                      int m, int n,
                      int start_row, int end_row,
                      int thread_id) {
        int n_row_blocks = m / QBLOCK_SIZE;
        for (int col = 0; col < n; col += 128) {
            int col_block = col / QBLOCK_SIZE;
            for (int row = start_row; row < end_row; row += 128) {
                int row_block = row / QBLOCK_SIZE;
                int block_id = row_block * n_row_blocks + col_block;

                q4f32s_qi8f32s_128x128_ukernel_otf(
                    w + (row * n / 2 + col / 2), n / 2,
                    s + block_id * QBLOCK_SIZE,
                    z + block_id * (QBLOCK_SIZE / 2),
                    in + col, in_scales[col / QBLOCK_SIZE],
                    out + row);
            }
        }
    };

    size_t n_threads = 4;
    vector<thread> threads(n_threads);

    int rows_per_thread = m / n_threads;
    assert(rows_per_thread % 128 == 0 && "Thread row blocks size must be divisible by 128");
    int start_row = 0;
    int end_row;
    for (int thread_id = 0; thread_id < n_threads; thread_id++) {
        end_row = start_row + rows_per_thread;
        threads[thread_id] = thread(worker, w, s, z, in, in_scales, out, m, n, start_row, end_row, thread_id);
        start_row += rows_per_thread;
    }
    for (auto& t : threads) {
        t.join();
    }
    threads.clear();
}
*/

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
