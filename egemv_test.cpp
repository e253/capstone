#include "common.hpp"
#include <cstring>
#include <iostream>

// llama ffn
void egemv_test1()
{
    int m = 4096;
    int n = 14336;

    uint8_t* w = (uint8_t*)_alloc(m * n / 2, 64);
    for (int i = 0; i < m * n / 2; i++) {
        w[i] = 0x55;
    }

    float* s = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++)
        s[i] = 2.0f;

    uint8_t* z = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        z[i] = 0x11;

    float* in = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        in[i] = 2.0f;

    float* out = (float*)_alloc(m * sizeof(float), 64);
    memset(out, 0, m * sizeof(float));

    q4f32s_egemv(
        w, s, z,
        in, out,
        m, n);

    bool passed = true;
    int printed = 0;
    int num_off = 0;
    for (int i = 0; i < m; i++) {
        if (out[i] != 229376.0f) {
            if (printed < 50) {
                std::cout << "Output[" << i << "] = " << out[i] << std::endl;
                printed++;
            }
            num_off++;
            passed = false;
        }
    }
    if (!passed) {
        std::cout << "Egemv Test 1 Failed" << std::endl;
        std::cout << "Number of Offenders: " << num_off << std::endl;
        exit(0);
    } else {
        std::cout << "Egemv Test 1 Passed" << std::endl;
    }
}

// phi-2 ffn
void egemv_test2()
{
    int m = 2560;
    int n = 10240;

    uint8_t* w = (uint8_t*)_alloc(m * n / 2, 64);
    for (int i = 0; i < m * n / 2; i++)
        w[i] = 0x55;

    float* s = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++)
        s[i] = 2.0f;

    uint8_t* z = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        z[i] = 0x11;

    float* in = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        in[i] = 2.0f;

    float* out = (float*)_alloc(m * sizeof(float), 64);
    memset(out, 0, m * sizeof(float));

    q4f32s_egemv(
        w, s, z,
        in, out,
        m, n);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (out[i] != 163840.0f) {
            if (i < 50) {
                std::cout << "Output[" << i << "] = " << out[i] << std::endl;
            }
            passed = false;
        }
    }
    if (!passed) {
        std::cout << "Egemv Test 1 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Egemv Test 2 Passed" << std::endl;
    }
}

void egemv_test3()
{
    int m = 4096;
    int n = 14336;

    uint8_t* w = (uint8_t*)_alloc(m * n / 2, 64);
    for (int i = 0; i < m * n / 2; i++)
        w[i] = 0x55;

    float* s = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int j = 0; j < m; j++) { // row idx
        for (int i = 0; i < n / QBLOCK_SIZE; i++) { // col idx
            if (i % 2 == 0) {
                s[i * m + j] = 1.0f;
            } else {
                s[i * m + j] = 2.0f;
            }
        }
    }

    uint8_t* z = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        z[i] = 0x11;

    float* in = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        in[i] = 2.0f;

    float* out = (float*)_alloc(m * sizeof(float), 64);
    memset(out, 0, m * sizeof(float));

    q4f32s_egemv(
        w, s, z,
        in, out,
        m, n);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (out[i] != 114688.0f) {
            if (i < 50) {
                std::cout << "Output[" << i << "] = " << out[i] << std::endl;
            }
            passed = false;
        }
    }
    if (!passed) {
        std::cout << "Egemv Test 3 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Egemv Test 3 Passed" << std::endl;
    }
}

int main()
{
    std::cout << "Egemv V" << VER << " Test" << std::endl;
    egemv_test1();
    egemv_test2();
    egemv_test3();
    std::cout << std::endl;
}