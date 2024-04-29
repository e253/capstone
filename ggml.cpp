#include <chrono>
#include <ggml/ggml-backend.h>
#include <ggml/ggml.h>
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
    ggml_tensor* gate_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 4096, 14336);
    random_init_array((char*)gate_proj->data, ggml_nbytes(gate_proj));
    ggml_tensor* down_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1, 14336, 4096);
    random_init_array((char*)gate_proj->data, ggml_nbytes(down_proj));

    ggml_tensor* up_proj_x = ggml_mul_mat(ctx, up_proj, x);
    ggml_tensor* gate_proj_x = ggml_mul_mat(ctx, gate_proj, x);
    ggml_tensor* y = ggml_mul_mat(ctx, down_proj, up_proj_x);

    const int NIT = 200;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < NIT; i++) {
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, /* n_threads = */ 4);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = 4096 * 14336 * 6 * (uint64_t)NIT;
    double flops_per_sec = flops_processed / sec;
    cout << "GFLOPS: " << flops_per_sec / 1e9 << endl;
    cout << endl;

    ggml_free(ctx);
}

int main()
{
    bench_llama_ffn();
}