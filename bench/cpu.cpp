#include "argparse/argparse.hpp"
#include "capstone/capstone.hpp"
#include <chrono>
#include <immintrin.h>
#include <iostream>

using namespace std;

void random_init_array(uint8_t* arr, int len)
{
    for (int i = 0; i < len; i++) {
        arr[i] = rand() % 128;
    }
}

void bench_llama_ffn()
{
    // up_proj 14336x4096
    // gate_proj 14336x4096
    // down proj 4096x14336
    // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
    // we do not do the elementwise multiply
    // we do not apply SwiGLU non-linearity in the hidden_dim, just the matmuls

    cout << "Benchmarking LLAMA FFN ..." << endl;
    cout << "down_proj @ (up_proj @ x * gate_proj @ x)" << endl;
    cout << "Hidden Dim: 14436, Dim: 4096" << endl;
    cout << endl;

    int dim = 4096;
    int hidden_dim = 14336;

    // arenas
    int w_byte_size = dim * hidden_dim / 2;
    int z_byte_size = hidden_dim * dim / QBLOCK_SIZE / 2;
    int s_byte_size = hidden_dim * dim / QBLOCK_SIZE * sizeof(float);

    q4f32s_tensor up_proj;
    up_proj.data = (uint8_t*)_mm_malloc(w_byte_size, 64);
    up_proj.z = (uint8_t*)_mm_malloc(z_byte_size, 64);
    up_proj.s = (float*)_mm_malloc(s_byte_size, 64);
    up_proj.m = hidden_dim; // m
    up_proj.n = dim; // n

    q4f32s_tensor gate_proj;
    gate_proj.data = (uint8_t*)_mm_malloc(w_byte_size, 64);
    gate_proj.z = (uint8_t*)_mm_malloc(z_byte_size, 64);
    gate_proj.s = (float*)_mm_malloc(s_byte_size, 64);
    gate_proj.m = hidden_dim;
    gate_proj.n = dim;

    q4f32s_tensor down_proj;
    down_proj.data = (uint8_t*)_mm_malloc(w_byte_size, 64);
    down_proj.z = (uint8_t*)_mm_malloc(z_byte_size, 64);
    down_proj.s = (float*)_mm_malloc(s_byte_size, 64);
    down_proj.m = dim;
    down_proj.n = hidden_dim;

    f32_vector x;
    x.data = (float*)_mm_malloc(dim * sizeof(float), 64);
    x.n = dim;

    qi8f32s_vector xq; // (hidden_dim,) but is also used for x(dim, )
    xq.data = (int8_t*)_mm_malloc(hidden_dim, 64);
    xq.s = (float*)_mm_malloc(hidden_dim / QBLOCK_SIZE * sizeof(float), 64);

    f32_vector s1;
    s1.data = (float*)_mm_malloc(hidden_dim * sizeof(float), 64);
    s1.n = hidden_dim;

    f32_vector s2;
    s2.data = (float*)_mm_malloc(hidden_dim * sizeof(float), 64);
    s2.n = hidden_dim;

    f32_vector y;
    y.data = (float*)_mm_malloc(hidden_dim * sizeof(float), 64);
    y.n = hidden_dim;

    const long long NIT = 200;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        q4f32s_qi8f32s_ffn(&up_proj, &gate_proj, &down_proj, &x, &xq, &y, &s1, &s2, 4);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    long long flops_processed = 4096 * 14336 * 6 * NIT;
    double GFLOPS = (double)flops_processed / sec * 1e-9;
    double BANDWIDTH = (double)(4096 * 14336 * 3) * .53375 * (double)NIT / sec * 1e-9;

    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    cout << "GFLOPS: " << GFLOPS << endl;
    cout << "BANDWIDTH: GB/s: " << BANDWIDTH << endl;
    cout << endl;

    // cleanup
    _mm_free(up_proj.data);
    _mm_free(up_proj.z);
    _mm_free(up_proj.s);

    _mm_free(gate_proj.data);
    _mm_free(gate_proj.z);
    _mm_free(gate_proj.s);

    _mm_free(down_proj.data);
    _mm_free(down_proj.z);
    _mm_free(down_proj.s);

    _mm_free(x.data);
    _mm_free(y.data);

    _mm_free(xq.data);
    _mm_free(xq.s);
    _mm_free(s1.data);
    _mm_free(s2.data);
}

void bench_phi_ffn()
{
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser parser("zig build bench_ggml --");
    parser.add_argument("--llama_ffn").flag().help("Bench LLAMA FFN");
    parser.add_argument("--phi_ffn").flag().help("Bench PHI FFN");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        cerr << err.what() << endl;
        cerr << parser << endl;
        cerr << "Invalid arguments. Exiting." << endl;
        exit(1);
    }

    if (parser["llama_ffn"] == true) {
        bench_llama_ffn();
    }
    // if (parser["llama_up_proj"] == true) {
    //     bench_llama_up_proj();
    // }
    if (parser["phi_ffn"] == true) {
        bench_phi_ffn();
    }
}