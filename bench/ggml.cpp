#include "ggml/ggml.h"
#include "argparse/argparse.hpp"
#include "ggml/ggml-backend.h"
#include <chrono>
#include <cstring>
#include <iostream>

using namespace std;

void random_init_array(char* arr, int len)
{
    for (int i = 0; i < len; i++) {
        arr[i] = rand() % 256;
    }
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

    size_t ctx_size = 0;
    {
        ctx_size += 4096 * 14336 * 3 * ggml_type_size(GGML_TYPE_Q4_1);
        ctx_size += (4096 + 2 * 14336) * ggml_type_size(GGML_TYPE_F32);
        ctx_size += 5 * ggml_tensor_overhead() + ggml_graph_overhead() + 1024;
    }

    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1);
    random_init_array((char*)x->data, ggml_nbytes(x));

    ggml_tensor* up_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 4096, 14336);
    random_init_array((char*)up_proj->data, ggml_nbytes(up_proj));

    ggml_tensor* gate_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 4096, 14336);
    random_init_array((char*)gate_proj->data, ggml_nbytes(gate_proj));

    ggml_tensor* down_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 14336, 4096);
    random_init_array((char*)gate_proj->data, ggml_nbytes(down_proj));

    ggml_tensor* up_proj_x = ggml_mul_mat(ctx, up_proj, x);
    ggml_tensor* gate_proj_x = ggml_mul_mat(ctx, gate_proj, x);
    ggml_tensor* y = ggml_mul_mat(ctx, down_proj, up_proj_x);

    const long long NIT = 200;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, /* n_threads = */ 4);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    long long flops_processed = 4096 * 14336 * 6 * NIT;
    double GFLOPS = (double)flops_processed / sec * 1e-9;
    double BANDWIDTH = (double)(4096 * 14336 * 3) * .5625 * double(NIT) / sec * 1e-9;
    // .625 bytes/value in q4_1

    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    cout << "GFLOPS: " << GFLOPS << endl;
    cout << "BANDWIDTH: GB/s: " << BANDWIDTH << endl;
    cout << endl;

    ggml_free(ctx);
}

void bench_phi_ffn()
{
    // down proj 4096x14336
    // gate_proj 14336x4096
    // up_proj 14336x4096
    // FFN(x) = down_proj @ (up_proj @ x * gate_proj @ x)
    // we do not do the elementwise multiply
    // we do not apply SwiGLU non-linearity in the hidden_dim
    // just the matmuls. skips 14k ops out of 176M total (4096 * 14336 * 3)

    cout << "Benchmarking PHI FFN ..." << endl;
    cout << "down_proj @ (up_proj @ x * gate_proj @ x)" << endl;
    cout << "Hidden Dim: 10240, Dim: 2560" << endl;
    cout << endl;

    size_t ctx_size = 0;
    {
        ctx_size += 2560 * 10240 * 3 * ggml_type_size(GGML_TYPE_Q4_1);
        ctx_size += (4096 + 2 * 14336) * ggml_type_size(GGML_TYPE_F32);
        ctx_size += 5 * ggml_tensor_overhead() + ggml_graph_overhead() + 1024;
    }

    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
    random_init_array((char*)x->data, ggml_nbytes(x));

    ggml_tensor* up_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 2560, 10240);
    random_init_array((char*)up_proj->data, ggml_nbytes(up_proj));

    ggml_tensor* gate_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 2560, 10240);
    random_init_array((char*)gate_proj->data, ggml_nbytes(gate_proj));

    ggml_tensor* down_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 10240, 2560);
    random_init_array((char*)gate_proj->data, ggml_nbytes(down_proj));

    ggml_tensor* up_proj_x = ggml_mul_mat(ctx, up_proj, x);
    ggml_tensor* gate_proj_x = ggml_mul_mat(ctx, gate_proj, x);
    ggml_tensor* y = ggml_mul_mat(ctx, down_proj, up_proj_x);

    const long long NIT = 200;

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, /* n_threads = */ 4);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    long long flops_processed = 2560 * 10240 * 6 * NIT;
    double GFLOPS = (double)flops_processed / sec * 1e-9;
    double BANDWIDTH = 2560 * 10240 * 3 / 32 * ggml_type_size(GGML_TYPE_Q4_1) * NIT / sec * 1e-9;

    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    cout << "GFLOPS: " << GFLOPS << endl;
    cout << "BANDWIDTH: GB/s: " << BANDWIDTH << endl;
    cout << endl;

    ggml_free(ctx);
}
/*
void bench_llama_up_proj()
{
    cout << "Benchmarking LLAMA up_proj ..." << endl;
    cout << "FFN(x): down_proj @ (up_proj @ x * gate_proj @ x)" << endl;
    cout << "Hidden Dim: 14436, Dim: 4096" << endl;
    cout << endl;

    size_t ctx_size = 0;
    {
        ctx_size += 4096 * 14336 * ggml_type_size(GGML_TYPE_Q4_1);
        ctx_size += 4096 * ggml_type_size(GGML_TYPE_F32);
        ctx_size += 3 * ggml_tensor_overhead() + ggml_graph_overhead() + 1024;
    }

    struct ggml_init_params params = {
        // 100 MB
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1);
    random_init_array((char*)x->data, ggml_nbytes(x));

    ggml_tensor* up_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 4096, 14336);
    random_init_array((char*)up_proj->data, ggml_nbytes(up_proj));

    ggml_tensor* out = ggml_mul_mat(ctx, up_proj, x);
    cout << "out type: "
         << "Q4_1" << endl;

    const int NIT = 200;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        // outputs in f32, offline quant
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = 4096 * 14336 * 2 * (uint64_t)NIT;
    double flops_per_sec = flops_processed / sec;
    cout << "GFLOPS: " << flops_per_sec / 1e9 << endl;
    cout << endl;

    ggml_free(ctx);
}
*/

int main(int argc, char** argv)
{
    cout << "NUMA_ENABLED: " << (ggml_is_numa() ? string("True") : string("False")) << endl
         << endl;

    argparse::ArgumentParser parser("zig build bench_ggml --");
    parser.add_argument("--llama_ffn").flag().help("Bench LLAMA FFN");
    // parser.add_argument("--llama_up_proj").flag().help("Bench LLAMA up_proj");
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
