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

// stackoverflow
#define FAST_ABS(x) _mm512_andnot_ps(_mm512_set1_ps(-0.0f), (x))

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
        __m512 f0 = _mm512_loadu_ps(in + j);
        __m512 f1 = _mm512_loadu_ps(in + j + 16);
        __m512 f2 = _mm512_loadu_ps(in + j + 32);
        __m512 f3 = _mm512_loadu_ps(in + j + 48);
        __m512 f4 = _mm512_loadu_ps(in + j + 64);
        __m512 f5 = _mm512_loadu_ps(in + j + 80);
        __m512 f6 = _mm512_loadu_ps(in + j + 96);
        __m512 f7 = _mm512_loadu_ps(in + j + 112);

        __m512 max1 = _mm512_max_ps(FAST_ABS(f0), FAST_ABS(f1));
        __m512 max2 = _mm512_max_ps(FAST_ABS(f2), FAST_ABS(f3));
        __m512 max3 = _mm512_max_ps(FAST_ABS(f4), FAST_ABS(f5));
        __m512 max4 = _mm512_max_ps(FAST_ABS(f6), FAST_ABS(f7));
        max1 = _mm512_max_ps(max1, max2);
        max3 = _mm512_max_ps(max3, max4);
        max1 = _mm512_max_ps(max1, max3);
        float max = _mm512_reduce_max_ps(max1);

        float scale = max > 127.0f ? (max / 127.0f) : 1.0f;

        out_s[j / QBLOCK_SIZE] = scale;

        // we need to order these 64+64 i32 values into 0, 2, 4, 6, ... 62, 1, 3, 5, ... 63
        int out_idx = j;
        for (int i = j; i < j + QBLOCK_SIZE / 2; i += 2) {
            out[out_idx] = static_cast<int8_t>(round(in[i] / scale));
            out_idx++;
        }
        for (int i = j + 1; i < j + QBLOCK_SIZE / 2; i += 2) {
            out[out_idx] = static_cast<int8_t>(round(in[i] / scale));
            out_idx++;
        }
        for (int i = j + QBLOCK_SIZE / 2; i < j + QBLOCK_SIZE; i += 2) {
            out[out_idx] = static_cast<int8_t>(round(in[i] / scale));
            out_idx++;
        }
        for (int i = j + QBLOCK_SIZE / 2 + 1; i < j + QBLOCK_SIZE; i += 2) {
            out[out_idx] = static_cast<int8_t>(round(in[i] / scale));
            out_idx++;
        }
        assert(out_idx == (j + QBLOCK_SIZE));

        /* Can't figure this out. There is probably a way to order them within avx512
        // we need to order these 64 i32 values into 0, 2, 4, 6, ... 30, 1, 3, 5, ... 31
        __m512i i0 = _mm512_cvtps_epi32(_mm512_div_round_ps(f0, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        __m512i i1 = _mm512_cvtps_epi32(_mm512_div_round_ps(f1, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); // 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        __m512i i2 = _mm512_cvtps_epi32(_mm512_div_round_ps(f2, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); // 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
        __m512i i3 = _mm512_cvtps_epi32(_mm512_div_round_ps(f3, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); // 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63

        i0 = _mm512_packs_epi32(i0, i1); // 0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23, 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31
        i2 = _mm512_packs_epi32(i2, i3); // 32, 33, 34, 35, 48, 49, 50, 51, 36, 37, 38, 39, 52, 53, 54, 55, 40, 41, 42, 43, 56, 57, 58, 59, 44, 45, 46, 47, 60, 61, 62, 63

        __m512i i4 = _mm512_cvtps_epi32(_mm512_div_round_ps(f4, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        __m512i i5 = _mm512_cvtps_epi32(_mm512_div_round_ps(f5, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); // 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        __m512i i6 = _mm512_cvtps_epi32(_mm512_div_round_ps(f6, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); // 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
        __m512i i7 = _mm512_cvtps_epi32(_mm512_div_round_ps(f7, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); // 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63

        _mm_storeu_epi8(out + j, _mm512_cvtepi32_epi8());
        _mm_storeu_epi8(out + j + 16, _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_div_round_ps(one, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
        _mm_storeu_epi8(out + j + 32, _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_div_round_ps(two, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
        _mm_storeu_epi8(out + j + 48, _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_div_round_ps(three, scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
        */
    }

    return 0;
}

// gets 64 u4 from memory and expands to u8
// those 64 u8 are given back as 0, 2, 4, 6, ..., 62, 1, 3, 5, ..., 63
static inline __m512i load_weights(const uint8_t* w)
{
    __m256i tmp = _mm256_loadu_si256((const __m256i*)w);
    __m512i weight = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_srli_epi32(tmp, 4)), tmp, 1);
    return _mm512_and_si512(weight, _mm512_set1_epi8(0x0F));
}

#define N_ITER 1 // can't be changed right now
#define ITER_LENGTH 512 // can be changed to multiple of 512
thread_ret_t q4f32s_qi8f32s_egemv_thread(void* _params)
{
    struct q4f32s_qi8f32s_egemv_params* params = (q4f32s_qi8f32s_egemv_params*)_params;

    uint8_t* w = params->w;
    float* s = params->s;
    uint8_t* z = params->z;
    int8_t* in = params->in;
    float* in_s = params->in_s;
    float* out = params->out;
    int m = params->m;
    int n = params->n;
    int start_row = params->tid * (m / params->n_threads);
    int end_row = start_row + (m / params->n_threads);

    assert(ITER_LENGTH % 128 == 0 && "ITER_LENGTH must be a multiple of 128");
    assert(N_ITER == 1 && "N_ITER must be 1");
    // printf("TID: %d - %d col_blocks\n", params->tid, n / ITER_LENGTH);
    for (int col_block = 0; col_block < n / ITER_LENGTH; col_block++) { // NOT QUANT BLOCK!
        for (int row = start_row; row < end_row; row += N_ITER) {
            float acc = 0;
            for (int col = col_block * ITER_LENGTH; col < (col_block + 1) * ITER_LENGTH; col += QBLOCK_SIZE) {
                int n_in_qblocks = n / QBLOCK_SIZE;
                int in_qblock = col / QBLOCK_SIZE;
                int out_qblock = row / QBLOCK_SIZE;
                int qblock_id = out_qblock * n_in_qblocks + in_qblock;
                // if (row == start_row) {
                //     printf("(TID: %d) - Processing row %d-%d, col %d-%d, qblock_id %d, col_block_id: %d\n", params->tid, start_row, end_row, col, col + QBLOCK_SIZE, qblock_id, col_block);
                // }

                __m512i acci = _mm512_setzero_si512();

                uint8_t _zero = z[(qblock_id * QBLOCK_SIZE + row % QBLOCK_SIZE) / 2];
                _zero >>= (!(row & 1) << 2); // 4 if row is even, 0 otherwise
                _zero &= 0x0F;
                __m512i zero = _mm512_set1_epi8(_zero);

                {
                    __m512i input = _mm512_loadu_epi8(&in[col]);
                    __m512i negative_input = _mm512_subs_epi8(_mm512_setzero_si512(), input);
                    __m512i weight = load_weights(&w[(row * n + col) / 2]);
                    // int sum = _mm512_reduce_add_epi32(weight); // weights good
                    // if (sum != 1347440720)
                    //     printf("sum: %d at (%d, %d): tid(%d)\n", sum, row, col, params->tid);
                    acci = _mm512_dpbusd_epi32(acci, weight, input);
                    acci = _mm512_dpbusd_epi32(acci, zero, negative_input);
                }
                {
                    __m512i input = _mm512_loadu_epi8(&in[col + 64]);
                    __m512i negative_input = _mm512_subs_epi8(_mm512_setzero_si512(), input);
                    __m512i weight = load_weights(&w[(row * n + col + 64) / 2]);
                    // int sum = _mm512_reduce_add_epi32(weight); // weights good
                    // if (sum != 1347440720)
                    //     printf("sum: %d at (%d, %d): tid(%d)\n", sum, row, col, params->tid);
                    acci = _mm512_dpbusd_epi32(acci, weight, input);
                    acci = _mm512_dpbusd_epi32(acci, zero, negative_input);
                }

                float combined_scale = s[qblock_id * QBLOCK_SIZE + row % QBLOCK_SIZE] * in_s[col / QBLOCK_SIZE];
                // if (combined_scale != 40000.0f) {
                //     printf("scale: %f at (%d, %d) qb(%d) off(%d): tid(%d)\n", combined_scale, row, col, qblock_id, qblock_id * QBLOCK_SIZE + row, params->tid);
                // }
                acc += _mm512_reduce_add_epi32(acci) * combined_scale;
            }
            out[row] += acc;
        }
    }

    return 0;
}

void q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    float* out,
    int m, int n,
    int n_threads)
{
    assert(m >= 512 && n >= 512 && "m and n must be at least 128");
    assert(m <= 32768 && n <= 32768 && "m and n can be at most 32768");
    assert(m % 512 == 0 && n % 512 == 0 && "m and n must be multiples of 512");

    vector<pthread_t> threads(n_threads - 1);
    vector<q4f32s_qi8f32s_egemv_params> params(n_threads);

    for (int i = 0; i < n_threads - 1; i++) {
        params[i].w = w;
        params[i].s = s;
        params[i].z = z;
        params[i].in = in;
        params[i].in_s = in_scales;
        params[i].out = out;
        params[i].m = m;
        params[i].n = n;
        params[i].tid = i;
        params[i].n_threads = n_threads;
        pthread_create(&threads[i], nullptr, q4f32s_qi8f32s_egemv_thread, (void*)&params[i]);
    }

    struct q4f32s_qi8f32s_egemv_params last_params = {
        w, s, z, in, in_scales, out, m, n, n_threads - 1, n_threads
    };
    q4f32s_qi8f32s_egemv_thread(&last_params);

    for (auto& t : threads) {
        pthread_join(t, nullptr);
    }

    threads.clear();
    params.clear();
}

void q4f32s_qi8f32s_ffn(
    q4f32s_tensor* up_proj,
    q4f32s_tensor* gate_proj,
    q4f32s_tensor* down_proj,
    f32_vector* x, qi8f32s_vector* xq,
    f32_vector* y,
    f32_vector* s1, f32_vector* s2,
    int n_threads)
{
    assert(up_proj->m == gate_proj->m && up_proj->n == gate_proj->n && up_proj->m == down_proj->n);
    assert(n_threads == 4 || n_threads == 2);

    vector<pthread_t> threads(n_threads - 1);
    vector<q4f32s_qi8f32s_ffn_params> params(n_threads);

    pthread_barrier_t op_barrier;
    pthread_barrier_init(&op_barrier, nullptr, n_threads);

    for (int i = 0; i < n_threads - 1; i++) {
        params[i].up_proj = up_proj;
        params[i].gate_proj = gate_proj;
        params[i].down_proj = down_proj;
        params[i].x = x;
        params[i].xq = xq;
        params[i].y = y;
        params[i].s1 = s1;
        params[i].s2 = s2;
        params[i].tid = i;
        params[i].n_threads = n_threads;
        params[i].op_barrier = &op_barrier;
        pthread_create(&threads[i], nullptr, q4f32s_qi8f32s_ffn_thread, (void*)&params[i]);
    }

    struct q4f32s_qi8f32s_ffn_params last_params = {
        up_proj,
        gate_proj,
        down_proj,
        x,
        xq,
        y,
        s1,
        s2,
        n_threads,
        n_threads - 1,
        &op_barrier,
    };
    q4f32s_qi8f32s_ffn_thread(&last_params);

    for (auto& t : threads) {
        pthread_join(t, nullptr);
    }

    pthread_barrier_destroy(&op_barrier);

    threads.clear();
    params.clear();
}

thread_ret_t q4f32s_qi8f32s_ffn_thread(void* _params)
{
    struct q4f32s_qi8f32s_ffn_params* params = (q4f32s_qi8f32s_ffn_params*)_params;

    q4f32s_tensor* up_proj = params->up_proj;
    // q4f32s_tensor* gate_proj = params->gate_proj;
    q4f32s_tensor* down_proj = params->down_proj;
    f32_vector* x = params->x;
    qi8f32s_vector* xq = params->xq;
    f32_vector* y = params->y;
    f32_vector* s1 = params->s1;
    f32_vector* s2 = params->s2;
    int n_threads = params->n_threads;
    int tid = params->tid;
    pthread_barrier_t* op_barrier = params->op_barrier;

    // Dequant(x) --> qi8 (4096,) --> up_proj (14336, 4096) --> s1 (14336,) --> xq (14336,) --> down_proj (14336, 4096) --> y (4096,)
    //                    --> gate_proj --> F32 (scratch2) (abandoned)

    // Q(x) --> xq
    struct f32_qi8f32s_params dequant_x_params = {
        x->data,
        xq->data,
        xq->s,
        x->n,
        tid,
        n_threads,
    };
    f32_qi8f32s_thread(&dequant_x_params);

    pthread_barrier_wait(op_barrier);

    // up_proj(xq) --> s1 (hidden_dim, )
    struct q4f32s_qi8f32s_egemv_params up_proj_params = {
        up_proj->data,
        up_proj->s,
        up_proj->z,
        xq->data, // (in)  dim
        xq->s,
        s1->data, // (out) hidden_dim
        up_proj->m, // (out) hidden_dim
        up_proj->n, // (in)  dim
        tid,
        n_threads,
    };
    q4f32s_qi8f32s_egemv_thread(&up_proj_params);

    // // gate_proj(xq) --> s2 (hidden_dim, )
    // struct q4f32s_qi8f32s_egemv_params gate_proj_params = {
    //     gate_proj->data,
    //     gate_proj->s,
    //     gate_proj->z,
    //     xq->data,
    //     xq->s,
    //     s2->data,
    //     gate_proj->m, // hidden_dim
    //     gate_proj->n, // dim
    //     tid,
    //     n_threads,
    // };
    // q4f32s_qi8f32s_egemv_thread(&gate_proj_params);

    pthread_barrier_wait(op_barrier);

    // Q(s2) --> xq (hidden_dim,)
    struct f32_qi8f32s_params quant_s2_params = {
        s2->data, // hidden_dim
        xq->data, // hidden_dim
        xq->s,
        s2->n, // hiddem_dim
        tid,
        n_threads,
    };
    f32_qi8f32s_thread(&quant_s2_params); // problems

    pthread_barrier_wait(op_barrier);

    struct q4f32s_qi8f32s_egemv_params down_proj_params = {
        down_proj->data,
        down_proj->s,
        down_proj->z,
        xq->data, // (in) hidden_dim
        xq->s,
        y->data,
        down_proj->m, // (in)  dim
        down_proj->n, // (out) hidden_dim
        tid,
        n_threads,
    };
    q4f32s_qi8f32s_egemv_thread(&down_proj_params);

    pthread_barrier_wait(op_barrier);

    return 0;
}
