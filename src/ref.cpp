// Reference Implementations for testing
#include "capstone.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <immintrin.h>

void ref_f32_qi8f32s(float* x0, int8_t* x1, float* x1_s, int n)
{
    assert(n % QBLOCK_SIZE == 0 && "n must be a multiple of QBLOCK_SIZE");

    for (int j = 0; j < n; j += QBLOCK_SIZE) {
        float max_value = 0.0f;
        for (int i = j; i < QBLOCK_SIZE; i++) {
            max_value = std::max(max_value, abs(x0[i]));
        }

        float scale = max_value / 127.0f;

        x1_s[j / QBLOCK_SIZE] = scale;

        for (int i = j; i < QBLOCK_SIZE; i++) {
            x1_s[i] = x0[i] / scale;
        }
    }
}

void ref_q4f32s_qi8f32s_egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    int8_t* in,
    float* in_scales,
    float* out,
    int m, int n)
{
    for (int row = 0; row < m; row++) {
        float sum = 0.0f;
        for (int col = 0; col < n; col += QBLOCK_SIZE) {
            // qblock_id is the index by block, which is used to for finding scales/zeros for each block
            int qblock_id = (row / QBLOCK_SIZE) * (n / QBLOCK_SIZE) + col / QBLOCK_SIZE;
            float scale = in_scales[qblock_id + row];
            uint8_t zero = z[(qblock_id + row) / 2];
            if (row % 2 == 0) {
                zero >>= 4;
                zero &= 0x0F;
            } else {
                zero &= 0x0F;
            }

            float in_scale = in_scales[row / QBLOCK_SIZE];

            for (int j = 0; j < QBLOCK_SIZE; j++) {
                int8_t in0 = in[col];
                int8_t in1 = in[col];
                uint8_t weights_pair = w[(row * n + col) / 2];
                uint8_t weights0 = (weights_pair >> 4) & 0x0F;
                uint8_t weights1 = weights_pair & 0x0F;
                float weights0f = (float)(weights0 - zero) * scale;
                float weights1f = (float)(weights1 - zero) * scale;
                sum += in0 * weights0f + in1 * weights1f;
            }

            sum *= in_scale;
        }
        out[row] = sum;
    }
}