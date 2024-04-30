#include <cstdint>

// nice potential api. leaving for now

enum CapDataType {
    Q4_F32S, // 4 bit weights, f32 scales, 4 bit zeros
    Q4_F16S,
    Q8_F32S,
    Q8_F16S,
    F16,
    F32,
};

enum CapBackend {
    CPU,
    CLSVM, // shared virtual memory, allocated for igpu
};

struct captensor {
    void* data;
    void* scale;
    void* zeros;
    int ndim;
    int dim[4];
    DataType dtype;
    Backend backend;
};

captensor* cap_new_tensor_2d(int rows, int cols, CapDataType dtype, CapBackend backend);

void cap_egemv(captensor* A, captensor* x, captensor* y);
void cap_ffn(captensor* up_proj, captensor* gate_proj, captensor* down_proj, captensor* x, captensor* y);