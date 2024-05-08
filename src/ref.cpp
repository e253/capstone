// Reference Implementations for testing
#include "capstone/capstone.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <limits>

using namespace std;

void ref_f32_qi8f32s(float* in, int8_t* out, float* out_s, int n)
{
    assert(n % QBLOCK_SIZE == 0 && "n must be a multiple of QBLOCK_SIZE");

    for (int j = 0; j < n; j += QBLOCK_SIZE) { // qblock
        float max_value = std::numeric_limits<float>::min();
        for (int i = j; i < j + QBLOCK_SIZE; i++) { // values
            max_value = max(max_value, abs(in[i]));
        }

        float scale = max_value > 127.0f ? (max_value / 127.0f) : 1.0f;

        out_s[j / QBLOCK_SIZE] = scale;

        for (int i = j; i < j + QBLOCK_SIZE; i++) { // values
            int8_t v = static_cast<int8_t>(round(in[i] / scale));
            out[i] = v;
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
    assert(512 <= n && n <= 32768 && 512 <= m && m <= 32768 && "n must be between 512 and 32768");
    assert(n % QBLOCK_SIZE == 0 && "n must be a multiple of QBLOCK_SIZE");

    for (int row = 0; row < m; row++) {
        float sum = 0.0f;
        for (int col = 0; col < n; col++) {
            // qblock_id is the index by block, which is used to for finding scales/zeros for each block
            int qblock_id = (row / QBLOCK_SIZE) * (n / QBLOCK_SIZE) + col / QBLOCK_SIZE;
            float scale = s[qblock_id * QBLOCK_SIZE + row % QBLOCK_SIZE];
            uint8_t zero = z[(qblock_id * QBLOCK_SIZE + row % QBLOCK_SIZE) / 2];
            if (row % 2 == 0) {
                zero >>= 4;
                zero &= 0x0F;
            } else {
                zero &= 0x0F;
            }

            int8_t input = in[col];
            uint8_t weight = w[(row * n + col) / 2];
            if (col % 2 == 0) {
                weight >>= 4;
                weight &= 0x0F;
            } else {
                weight &= 0x0F;
            }

            float in_scale = in_scales[col / QBLOCK_SIZE];

            sum += ((float)(weight - zero) * scale) * (input * in_scale);
            // sum += (float)((int)((short)weight * (short)input - (short)zero * (short)input)) * scale * in_scale; no accuracy difference
        }
        out[row] = sum;
    }
}
