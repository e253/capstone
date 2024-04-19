#include "common.hpp"
#include <chrono>
#include <iostream>

using namespace std;

int main()
{
    cout << "Egemv V" << VER << " Bench" << endl;

    int m = 4096;
    int n = 14336;

    uint8_t* w = (uint8_t*)_alloc(m * n / 2, 64);
    float* s = (float*)_alloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)_alloc(m * n / QBLOCK_SIZE / 2, 64);
    float* in = (float*)_alloc(n * sizeof(float), 64);
    float* out = (float*)_alloc(m * sizeof(float), 64);

    const uint64_t NIT = 500;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        q4f32s_egemv(
            w, s, z,
            in, out,
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

    _free(w);
    _free(s);
    _free(z);
    _free(in);
    _free(out);
}