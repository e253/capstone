#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#define QBLOCK_SIZE 128

using namespace std;

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
#define REDUCE_ADD(acc) _mm512_reduce_add_ps(_mm512_cvtepi32_ps(acc))
#else
typedef __m512 acc_t;
#define REDUCE_ADD(acc) _mm512_reduce_add_ps((acc))
#endif

#define CLAMP(x, lo, hi) (x < lo ? lo : (x > hi ? hi : x))

inline acc_t mul_input_weight_accum(__m512i input, __m512i weight, __m512i zero, acc_t acc)
{
    /*
    VNNI likes u8 * i8 multiplications
    input(i8) * (weights (u8) - zeros (u8)) == weights(u8)*input(i8) - zeros(u8)*input(i8)
    Signed muliplications are possible, but only with ymm regs ^_^
    */

#ifdef I32ACCUM
    acc = _mm512_dpbusds_epi32(acc, weight, input);
    __m512i tmp = _mm512_dpbusds_epi32(_mm512_setzero_si512(), zero, input);
    acc = _mm512_sub_epi32(acc, tmp);
    return acc;
#else
    __m512i tmp1 = _mm512_dpbusds_epi32(_mm512_setzero_epi32(), weight, input);
    __m512i tmp2 = _mm512_dpbusds_epi32(_mm512_setzero_epi32(), zero, input);
    tmp1 = _mm512_sub_epi32(tmp1, tmp2);
    acc = _mm512_add_ps(acc, _mm512_cvtepi32_ps(tmp1));
    return acc;
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
void q4f32s_qi8f32s_128x128_ukernel_offline(
    uint8_t* __restrict w,
    uint64_t w_rs,
    float* __restrict scales,
    uint8_t* __restrict zeros,
    int8_t* __restrict in,
    float in_scale,
    int8_t* __restrict out,
    float out_scale)
{

    float io_scale = in_scale / out_scale;

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
        // CLAMP makes sure the float is within the range of int8_t
        out[row] += (int8_t)CLAMP((REDUCE_ADD(acc1) * scales[row] * io_scale), -128.0f, 127.0f);
        out[row + 1] += (int8_t)CLAMP((REDUCE_ADD(acc2) * scales[row + 1] * io_scale), -128.0f, 127.0f);
        out[row + 2] += (int8_t)CLAMP((REDUCE_ADD(acc3) * scales[row + 2] * io_scale), -128.0f, 127.0f);
        out[row + 3] += (int8_t)CLAMP((REDUCE_ADD(acc4) * scales[row + 3] * io_scale), -128.0f, 127.0f);
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
    assert(m % 128 == 0 && "Row size must be divisble by 128");
    assert(n % QBLOCK_SIZE == 0 && "Col size must be divisble by 128");

    auto process_128_rows_n_cols = [&](uint8_t* w, float* s, uint8_t* z,
                                       int8_t* in, float* in_scales,
                                       int8_t* out, float* out_scales,
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

                q4f32s_qi8f32s_128x128_ukernel_offline(
                    w + (row * n / 2 + col / 2), n / 2,
                    s + block_id * QBLOCK_SIZE,
                    z + block_id * (QBLOCK_SIZE / 2),
                    in + col, in_scales[col / QBLOCK_SIZE],
                    out + row, out_scales[row / QBLOCK_SIZE]);
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
        threads[thread_id] = thread(process_128_rows_n_cols, w, s, z, in, in_scales, out, out_scales, m, n, start_row, end_row, thread_id);
#ifndef _WIN32
/*
        // https://eli.thegreenplace.net/2016/c11-threads-affinity-and-hyperthreading/
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        int rc = pthread_setaffinity_np(threads[thread_id].native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            cerr << "Error calling pthread_setaffinity_np: " << rc << endl;
            exit(0);
        }
*/
#endif
        start_row += rows_per_thread;
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
*/
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
        // CLAMP makes sure the float is within the range of int8_t
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

    auto process_128_rows_n_cols = [&](uint8_t* w, float* s, uint8_t* z,
                                       int8_t* in, float* in_scales,
                                       float* out,
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
#ifndef _WIN32
    cpu_set_t cpuset;
#endif

    int rows_per_thread = m / n_threads;
    assert(rows_per_thread % 128 == 0 && "Thread row blocks size must be divisible by 128");
    int start_row = 0;
    int end_row;

    for (int thread_id = 0; thread_id < n_threads; thread_id++) {
        end_row = start_row + rows_per_thread;
        threads[thread_id] = thread(process_128_rows_n_cols, w, s, z, in, in_scales, out, m, n, start_row, end_row, thread_id);
#ifndef _WIN32
/*
        // https://eli.thegreenplace.net/2016/c11-threads-affinity-and-hyperthreading/
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        int rc = pthread_setaffinity_np(threads[thread_id].native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            cerr << "Error calling pthread_setaffinity_np: " << rc << endl;
            exit(0);
        }
*/
#endif
        start_row += rows_per_thread;
    }
    for (auto& t : threads) {
        t.join();
    }
    threads.clear();
}

// benchmarks activated with -DBENCH
#define TESTBENCH 1
#ifdef TESTBENCH
void test_1()
{
    int m = 128;
    int n = 128;

    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);
    for (int i = 0; i < m * n / 2; i++)
        w[i] = 0x55;

    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++)
        s[i] = 2.0f;

    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        z[i] = 0x11;

    int8_t* in = (int8_t*)_mm_malloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        in[i] = 2;
    float in_s = 1.0f;

    int8_t* out = (int8_t*)_mm_malloc(m, 64);
    std::memset(out, 0, m);
    float out_s = 20.48f;
    // keeps 2048 i32 into -128 <--> 127

    q4f32s_qi8f32s_128x128_ukernel_offline(
        w, m / 2,
        s, z,
        in, in_s,
        out, out_s);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (out[i] != 100) {
            std::cout << "Output[" << i << "] = " << (int)out[i] << std::endl;
            passed = false;
        }
    }

    if (!passed) {
        std::cout << std::endl;
        std::cout << "Kernel Test 1 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Kernel Test 1 Passed" << std::endl;
    }

    _mm_free(w);
    _mm_free(s);
    _mm_free(z);
    _mm_free(in);
    _mm_free(out);
}
void test_2()
{
    int m = 128;
    int n = 128;

    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);
    for (int i = 0; i < m * n / 2; i++)
        w[i] = 0x55;

    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++)
        s[i] = 2.0f;

    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        z[i] = 0x11;

    int8_t* in = (int8_t*)_mm_malloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        in[i] = 2;
    float in_s = 1.0f;

    float* out = (float*)_mm_malloc(m * sizeof(float), 64);
    std::memset(out, 0, m);
    // keeps 2048 i32 into -128 <--> 127

    q4f32s_qi8f32s_128x128_ukernel_otf(
        w, m / 2,
        s, z,
        in, in_s,
        out);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (out[i] != 2048.0f) {
            std::cout << "Output[" << i << "] = " << (int)out[i] << std::endl;
            passed = false;
        }
    }

    if (!passed) {
        std::cout << std::endl;
        std::cout << "Kernel Test 2 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Kernel Test 2 Passed" << std::endl;
    }

    _mm_free(w);
    _mm_free(s);
    _mm_free(z);
    _mm_free(in);
    _mm_free(out);
}

#include <chrono>
void bench_offline_500_for_dim(int m, int n)
{
    assert(m % 128 == 0 && "Row size must be divisble by 128");
    assert(n % 128 == 0 && "Col size must be divisble by 128");

    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);
    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)_mm_malloc(n, 64);
    float* in_scales = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);
    int8_t* out = (int8_t*)_mm_malloc(m, 64);
    float* out_scales = (float*)_mm_malloc(m / QBLOCK_SIZE * sizeof(float), 64);

    const uint64_t NIT = 500;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        q4f32s_qi8f32s_egemv_offline(
            w, s, z,
            in, in_scales,
            out, out_scales,
            m, n);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = m * n * 2 * NIT;
    double flops_sec = flops_processed / sec;
    double gflops = flops_sec / (1e9);
    cout << "GFLOPS: " << gflops << endl;
    cout << endl;

    _mm_free(w);
    _mm_free(s);
    _mm_free(z);
    _mm_free(in);
    _mm_free(in_scales);
    _mm_free(out);
    _mm_free(out_scales);
}
void bench_otf_500_for_dim(int m, int n)
{
    assert(m % 128 == 0 && "Row size must be divisble by 128");
    assert(n % 128 == 0 && "Col size must be divisble by 128");

    uint8_t* w = (uint8_t*)_mm_malloc(m * n / 2, 64);
    float* s = (float*)_mm_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)_mm_malloc(m * n / QBLOCK_SIZE / 2, 64);
    int8_t* in = (int8_t*)_mm_malloc(n, 64);
    float* in_scales = (float*)_mm_malloc(n / QBLOCK_SIZE * sizeof(float), 64);
    float* out = (float*)_mm_malloc(m * sizeof(float), 64);

    const uint64_t NIT = 500;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        q4f32s_qi8f32s_egemv_otf(
            w, s, z,
            in, in_scales,
            out,
            m, n);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = m * n * 2 * NIT;
    double flops_sec = flops_processed / sec;
    double gflops = flops_sec / (1e9);
    cout << "GFLOPS: " << gflops << endl;
    cout << endl;

    _mm_free(w);
    _mm_free(s);
    _mm_free(z);
    _mm_free(in);
    _mm_free(in_scales);
    _mm_free(out);
}
int main()
{
    cout << "Testing" << endl;
    test_1();

    cout << "Llama FNN (offline)" << endl;
    bench_offline_500_for_dim(4096, 14336);
    cout << "Llama FNN (otf)" << endl;
    bench_otf_500_for_dim(4096, 14336);

    cout << "Phi-2 FFN (offline)" << endl;
    bench_offline_500_for_dim(2560, 10240);
    cout << "Phi-1 FFN (otf)" << endl;
    bench_otf_500_for_dim(2560, 10240);
    return 0;
}
#endif
