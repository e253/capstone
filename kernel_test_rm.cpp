#include "common.hpp"
#include <cstring>
#include <iostream>

void kernel_test1()
{
    int m = 128;
    int n = 128;

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

    q4f32s_128x128_ukernel(
        Weights, n / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        Output);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (Output[i] != 2048.0f) {
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
    int n = 128;

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

    q4f32s_128x128_ukernel(
        Weights, n / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        Output);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (Output[i] != 1024.0f) {
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
    int n = 128;

    uint8_t* Weights = (uint8_t*)_alloc(m * n / 2, 64);
    for (int row = 0; row < m; row++) { // row idx
        for (int col = 0; col < n / 2; col++) { // col idx
            if (col % 2 == 0) {
                Weights[row * n / 2 + col] = 0x33;
            } else {
                Weights[row * n / 2 + col] = 0x55;
            }
        }
    }

    float* W_Scales = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int row = 0; row < m; row++) { // row idx
        for (int col = 0; col < n / QBLOCK_SIZE; col++) { // col idx
            if (col % 2 == 0) {
                W_Scales[col * m + row] = 1.0f;
            } else {
                W_Scales[col * m + row] = 2.0f;
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

    q4f32s_128x128_ukernel(
        Weights, n / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        Output);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (Output[i] != 768.0f) {
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
    int n = 128;

    uint8_t* Weights = (uint8_t*)_alloc(m * n / 2, 64);
    for (int row = 0; row < m; row++) { // row idx
        for (int col = 0; col < n / 2; col++) { // col idx
            if (row % 2 == 0) {
                Weights[row * n / 2 + col] = 0x33;
            } else {
                Weights[row * n / 2 + col] = 0x55;
            }
        }
    }

    float* W_Scales = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++) // row idx
        W_Scales[i] = 2.0f;

    uint8_t* W_Zeros = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        W_Zeros[i] = 0x11;

    float* Input = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        Input[i] = 2.0f;

    float* Output = (float*)_alloc(m * sizeof(float), 64);
    std::memset(Output, 0, m * sizeof(float));

    q4f32s_128x128_ukernel(
        Weights, n / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        Output);

    bool passed = true;
    for (int i = 0; i < m; i += 4) {
        if (Output[i] != 1024.0f && Output[i + 1] != 1024.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 1 << "] = " << Output[i] << std::endl;
            passed = false;
        }
        if (Output[i + 2] != 2048.0f && Output[i + 3] != 2048.0f) {
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
    int m = 256;
    int n = 256;

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

    // use lower 128x128 square
    q4f32s_128x128_ukernel(
        // ptr + row * row_stride + col / 2
        Weights + 128 * n / 2 + 128 / 2, n / 2,
        // ptr + (col / QBLOCK_SIZE) * col_stride + (row)
        W_Scales + (128 / QBLOCK_SIZE) * m + 128, m,
        // ptr + (col / QBLOCK_SIZE / 2) * col_stride + row / 2
        W_Zeros + (128 / QBLOCK_SIZE / 2) * m / 2 + 128 / 2, m / 2,
        // col
        Input + 128,
        // row
        Output + 128);

    bool passed = true;
    for (int i = 128; i < m; i += 4) {
        if (Output[i] != 1024.0f && Output[i + 1] != 1024.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 1 << "] = " << Output[i] << std::endl;
            passed = false;
        }
        if (Output[i + 2] != 2048.0f && Output[i + 3] != 2048.0f) {
            std::cout << "Output[" << i + 2 << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 3 << "] = " << Output[i] << std::endl;
            passed = false;
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

/*
void kernel_test6()
{
    int m = 128;
    int n = 4096;

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
        Output,
        n);
    q4f32s_ukernel_epiloque(Output);

    bool passed = true;
    for (int i = 0; i < m; i += 4) {
        if (Output[i] != 32768.0f && Output[i + 1] != 32768.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 1 << "] = " << Output[i] << std::endl;
            passed = false;
        }
        if (Output[i + 2] != 65536.0f && Output[i + 3] != 65536.0f) {
            std::cout << "Output[" << i + 2 << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 3 << "] = " << Output[i] << std::endl;
            passed = false;
        }
    }

    if (!passed) {
        std::cout << std::endl;
        std::cout << "Kernel Test 6 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Kernel Test 6 Passed" << std::endl;
    }

    _free(Weights);
    _free(W_Scales);
    _free(W_Zeros);
    _free(Input);
    _free(Output);
}

void kernel_test7()
{
    int m = 384;
    int n = 4096;

    uint8_t* Weights = (uint8_t*)_alloc(m * n / 2, 64);
    for (int j = 64; j < m / 2; j++) { // row idx
        for (int i = 0; i < n; i++) { // col idx
            if (j % 2 == 0) {
                Weights[i * m / 2 + j] = 0x33;
            } else {
                Weights[i * m / 2 + j] = 0x55;
            }
        }
    }

    float* W_Scales = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int j = 128; j < m; j++) { // row idx
        for (int i = 0; i < n / QBLOCK_SIZE; i++) { // col idx
            W_Scales[i * m + j] = 2.0f;
        }
    }

    uint8_t* W_Zeros = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 64; i < m * n / QBLOCK_SIZE / 2; i++)
        W_Zeros[i] = 0x11;

    float* Input = (float*)_alloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        Input[i] = 2.0f;

    float* Output = (float*)_alloc(m * sizeof(float), 64);
    std::memset(Output, 0, m * sizeof(float));

    q4f32s_ukernel_prelude();
    q4f32s_ukernel(
        Weights + 256 / 2, m / 2,
        W_Scales + 256, m,
        W_Zeros + 256 / 2, m / 2,
        Input,
        Output + 256,
        n);
    q4f32s_ukernel_epiloque(Output + 256);

    bool passed = true;
    for (int i = 256; i < m; i += 4) {
        if (Output[i] != 32768.0f && Output[i + 1] != 32768.0f) {
            std::cout << "Output[" << i << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 1 << "] = " << Output[i] << std::endl;
            passed = false;
        }
        if (Output[i + 2] != 65536.0f && Output[i + 3] != 65536.0f) {
            std::cout << "Output[" << i + 2 << "] = " << Output[i] << std::endl;
            std::cout << "Output[" << i + 3 << "] = " << Output[i] << std::endl;
            passed = false;
        }
    }

    if (!passed) {
        std::cout << std::endl;
        std::cout << "Kernel Test 7 Failed" << std::endl;
        exit(0);
    } else {
        std::cout << "Kernel Test 7 Passed" << std::endl;
    }

    _free(Weights);
    _free(W_Scales);
    _free(W_Zeros);
    _free(Input);
    _free(Output);
}
*/
int main()
{
    std::cout << "Kenel V" << KERNEL_VER << " Test" << std::endl;
    kernel_test1();
    kernel_test2();
    kernel_test3();
    kernel_test4();
    kernel_test5();
    // kernel_test6();
    // kernel_test7();
    std::cout << std::endl;
    return 0;
}