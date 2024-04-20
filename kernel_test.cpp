#include "common.hpp"
#include <cstring>
#include <iostream>

void kernel_test1()
{
    int m = 128;
    int n = 512;

    uint8_t* Weights = (uint8_t*)_alloc(m * n / 2, 64);
    for (int i = 0; i < m * n / 2; i++)
        Weights[i] = 0x55;

    float* W_Scales = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++)
        W_Scales[i] = 2.0f;

    uint8_t* W_Zeros = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        W_Zeros[i] = 0x11;

    float* Input = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        Input[i] = 2.0f;

    float* Output = (float*)_alloc(m * sizeof(float), 64);
    std::memset(Output, 0, m * sizeof(float));

    q4f32s_ukernel_prelude();
    q4f32s_ukernel(
        Weights, m / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        nullptr,
        n);
    q4f32s_ukernel_epiloque(Output);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (Output[i] != 8192.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
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

    _free(Weights);
    _free(W_Scales);
    _free(W_Zeros);
    _free(Input);
    _free(Output);
}

void kernel_test2()
{
    int m = 128;
    int n = 512;

    uint8_t* Weights = (uint8_t*)_alloc(m * n / 2, 64);
    for (int i = 0; i < m * n / 2; i++)
        Weights[i] = 0x55;

    float* W_Scales = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int j = 0; j < m; j++) { // row idx
        for (int i = 0; i < n / QBLOCK_SIZE; i++) { // col idx
            if (i % 2 == 0) {
                W_Scales[i * m + j] = 1.0f;
            } else {
                W_Scales[i * m + j] = 2.0f;
            }
        }
    }

    uint8_t* W_Zeros = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        W_Zeros[i] = 0x11;

    float* Input = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        Input[i] = 2.0f;

    float* Output = (float*)_alloc(m * sizeof(float), 64);
    std::memset(Output, 0, m * sizeof(float));

    q4f32s_ukernel_prelude();
    q4f32s_ukernel(
        Weights, m / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        Output,
        n);
    q4f32s_ukernel_epiloque(Output);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (Output[i] != 4096.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
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

    _free(Weights);
    _free(W_Scales);
    _free(W_Zeros);
    _free(Input);
    _free(Output);
}

void kernel_test3()
{
    int m = 128;
    int n = 512;

    uint8_t* Weights = (uint8_t*)_alloc(m * n / 2, 64);
    for (int j = 0; j < m / 2; j++) { // row idx
        for (int i = 0; i < n; i++) { // col idx
            if (i % 2 == 0) {
                Weights[i * m / 2 + j] = 0x33;
            } else {
                Weights[i * m / 2 + j] = 0x55;
            }
        }
    }

    float* W_Scales = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int j = 0; j < m; j++) { // row idx
        for (int i = 0; i < n / QBLOCK_SIZE; i++) { // col idx
            if (i % 2 == 0) {
                W_Scales[i * m + j] = 1.0f;
            } else {
                W_Scales[i * m + j] = 2.0f;
            }
        }
    }

    uint8_t* W_Zeros = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        W_Zeros[i] = 0x11;

    float* Input = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        Input[i] = 2.0f;

    float* Output = (float*)_alloc(m * sizeof(float), 64);
    std::memset(Output, 0, m * sizeof(float));

    q4f32s_ukernel_prelude();
    q4f32s_ukernel(
        Weights, m / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        nullptr,
        n);
    q4f32s_ukernel_epiloque(Output);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (Output[i] != 3072.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
            passed = false;
        }
    }
    if (!passed) {
        std::cout << std::endl;
        std::cout << "Kernel Test 3 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Kernel Test 3 Passed" << std::endl;
    }

    _free(Weights);
    _free(W_Scales);
    _free(W_Zeros);
    _free(Input);
    _free(Output);
}

void kernel_test4()
{
    int m = 128;
    int n = 512;

    uint8_t* Weights = (uint8_t*)_alloc(m * n / 2, 64);
    for (int j = 0; j < m / 2; j++) { // row idx
        for (int i = 0; i < n; i++) { // col idx
            if (j % 2 == 0) {
                Weights[i * m / 2 + j] = 0x33;
            } else {
                Weights[i * m / 2 + j] = 0x55;
            }
        }
    }

    float* W_Scales = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int j = 0; j < m; j++) { // row idx
        for (int i = 0; i < n / QBLOCK_SIZE; i++) { // col idx
            W_Scales[i * m + j] = 2.0f;
        }
    }

    uint8_t* W_Zeros = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        W_Zeros[i] = 0x11;

    float* Input = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        Input[i] = 2.0f;

    float* Output = (float*)_alloc(m * sizeof(float), 64);
    std::memset(Output, 0, m * sizeof(float));

    q4f32s_ukernel_prelude();
    q4f32s_ukernel(
        Weights, m / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        nullptr,
        n);
    q4f32s_ukernel_epiloque(Output);

    bool passed = true;
    for (int i = 0; i < m; i += 4) {
        if (Output[i] != 4096.0f && Output[i + 1] != 4096.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 1 << "] = " << Output[i] << std::endl;
            passed = false;
        }
        if (Output[i + 2] != 8192.0f && Output[i + 3] != 8192.0f) {
            std::cout << "Output[" << i + 2 << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 3 << "] = " << Output[i] << std::endl;
        }
    }
    if (!passed) {
        std::cout << std::endl;
        std::cout << "Kernel Test 4 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Kernel Test 4 Passed" << std::endl;
    }

    _free(Weights);
    _free(W_Scales);
    _free(W_Zeros);
    _free(Input);
    _free(Output);
}

void kernel_test5()
{
    int m = 128;
    int n = 1024;

    uint8_t* Weights = (uint8_t*)_alloc(m * n / 2, 64);
    for (int j = 0; j < m / 2; j++) { // row idx
        for (int i = 0; i < n; i++) { // col idx
            if (j % 2 == 0) {
                Weights[i * m / 2 + j] = 0x33;
            } else {
                Weights[i * m / 2 + j] = 0x55;
            }
        }
    }

    float* W_Scales = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int j = 0; j < m; j++) { // row idx
        for (int i = 0; i < n / QBLOCK_SIZE; i++) { // col idx
            W_Scales[i * m + j] = 2.0f;
        }
    }

    uint8_t* W_Zeros = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        W_Zeros[i] = 0x11;

    float* Input = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        Input[i] = 2.0f;

    float* Output = (float*)_alloc(m * sizeof(float), 64);
    std::memset(Output, 0, m * sizeof(float));

    q4f32s_ukernel_prelude();
    q4f32s_ukernel(
        Weights, m / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        nullptr,
        n);
    q4f32s_ukernel_epiloque(Output);

    bool passed = true;
    for (int i = 0; i < m; i += 4) {
        if (Output[i] != 8192.0f && Output[i + 1] != 8192.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 1 << "] = " << Output[i] << std::endl;
            passed = false;
        }
        if (Output[i + 2] != 16384.0f && Output[i + 3] != 16384.0f) {
            std::cout << "Output[" << i + 2 << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 3 << "] = " << Output[i] << std::endl;
        }
    }
    if (!passed) {
        std::cout << std::endl;
        std::cout << "Kernel Test 5 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Kernel Test 5 Passed" << std::endl;
    }

    _free(Weights);
    _free(W_Scales);
    _free(W_Zeros);
    _free(Input);
    _free(Output);
}

int main()
{
    std::cout << "Kenel V" << KERNEL_VER << " Test" << std::endl;
    kernel_test1();
    kernel_test2();
    kernel_test3();
    kernel_test4();
    kernel_test5();
    std::cout << std::endl;
    return 0;
}