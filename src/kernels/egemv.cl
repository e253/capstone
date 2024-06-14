inline char clamp(float x)
{
    return (char)(x > 127.0f ? 127 : (x < -128.0f ? -128 : round(x)));
}

inline char get0(uchar w)
{
    return (char)((w >> 4) & 0x0F);
}

inline char get1(uchar w)
{
    return (char)(w & 0x0F);
}

__kernel void f32_qi8f32s(
    __global float *restrict in,
    __global char *restrict out,
    __global float *restrict out_s,
    int n,
    int n_el_per_thread)
{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);

    float max = fabs(in[gid * n_el_per_thread]);
    for (int i = gid * n_el_per_thread; i < (gid + 1) * n_el_per_thread; i++)
        max = fmax(fabs(max), fabs(in[i]));

    max = work_group_reduce_max(max);

    __local float _scale; // poor mans work_group_broadcast
    float scale;
    if (lid == 0)
    {
        scale = max > 127.0f ? max / 127.0f : 1.0f;
        out_s[get_group_id(0)] = scale;
        _scale = scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    scale = _scale;

    for (uint i = gid * n_el_per_thread; i < (gid + 1) * n_el_per_thread; i++)
    {
        out[i] = clamp(round(in[i] / scale));
    }
}

__kernel void q4f32s_qi8f32s_egemv_kernel(
    __global uchar4 *restrict w,
    __global float *restrict s,
    __global uchar *restrict z,
    __global char8 *restrict in,
    __global float *restrict in_scales,
    __global float *restrict out,
    int m, int n, int n_blocks_per_thread)
{
    const int QBLOCK_SIZE = 128;
    const int row_sz2_block = get_local_id(1);
    const int out_qblock = get_global_id(2);

    float acc1 = 0;
    float acc2 = 0;

    for (uint qblock = 0; qblock < n_blocks_per_thread; qblock++)
    {
        // qblock-acc
        float _acc1 = 0;
        float _acc2 = 0;
        const int in_qblock = get_local_id(0) * n_blocks_per_thread + qblock;
        const int QBLOCK_ID = out_qblock * (n / QBLOCK_SIZE) + in_qblock;

        // Set Zeros
        uchar _zero1 = z[QBLOCK_ID * QBLOCK_SIZE / 2 + row_sz2_block * 2 / 2];
        uchar _zero2 = _zero1;
        char zero1 = (char)((_zero1 >> 4) & 0x0F);
        char zero2 = (char)(_zero2 & 0x0F);

        for (uint i = 0; i < 128; i += 8)
        {
            // Load Input
            char8 input = in[(in_qblock * 128 + i) / 8];

            // logical row = out_qblock * QBLOCK_SIZE + row_sz2_block * 2
            // logical col = in_qblock * QBLOCK_SIZE + i
            // each is divided by 8 to get the index. 2 values per byte, 4 bytes per index
            uchar4 weights1 = w[((out_qblock * QBLOCK_SIZE + row_sz2_block * 2) * n + in_qblock * QBLOCK_SIZE + i) / 8];

            _acc1 = mad(convert_float(get0(weights1.s0) - zero1), convert_float(input.s0), _acc1);
            _acc1 = mad(convert_float(get1(weights1.s0) - zero1), convert_float(input.s1), _acc1);
            _acc1 = mad(convert_float(get0(weights1.s1) - zero1), convert_float(input.s2), _acc1);
            _acc1 = mad(convert_float(get1(weights1.s1) - zero1), convert_float(input.s3), _acc1);
            _acc1 = mad(convert_float(get0(weights1.s2) - zero1), convert_float(input.s4), _acc1);
            _acc1 = mad(convert_float(get1(weights1.s2) - zero1), convert_float(input.s5), _acc1);
            _acc1 = mad(convert_float(get0(weights1.s3) - zero1), convert_float(input.s6), _acc1);
            _acc1 = mad(convert_float(get1(weights1.s3) - zero1), convert_float(input.s7), _acc1);

            // logical row = out_qblock * QBLOCK_SIZE + row_sz2_block * 2
            // logical col = in_qblock * QBLOCK_SIZE + i ** + 1 **
            // each is divided by 8 to get the index. 2 values per byte, 4 bytes per index
            uchar4 weights2 = w[((out_qblock * QBLOCK_SIZE + row_sz2_block * 2 + 1) * n + in_qblock * QBLOCK_SIZE + i) / 8];

            _acc2 = mad(convert_float(get0(weights2.s0) - zero2), convert_float(input.s0), _acc2);
            _acc2 = mad(convert_float(get1(weights2.s0) - zero2), convert_float(input.s1), _acc2);
            _acc2 = mad(convert_float(get0(weights2.s1) - zero2), convert_float(input.s2), _acc2);
            _acc2 = mad(convert_float(get1(weights2.s1) - zero2), convert_float(input.s3), _acc2);
            _acc2 = mad(convert_float(get0(weights2.s2) - zero2), convert_float(input.s4), _acc2);
            _acc2 = mad(convert_float(get1(weights2.s2) - zero2), convert_float(input.s5), _acc2);
            _acc2 = mad(convert_float(get0(weights2.s3) - zero2), convert_float(input.s6), _acc2);
            _acc2 = mad(convert_float(get1(weights2.s3) - zero2), convert_float(input.s7), _acc2);
        } // block process

        acc1 += (float)_acc1 * s[QBLOCK_ID * QBLOCK_SIZE + row_sz2_block * 2] * in_scales[in_qblock];
        acc2 += (float)_acc2 * s[QBLOCK_ID * QBLOCK_SIZE + row_sz2_block * 2 + 1] * in_scales[in_qblock];
    } // qblock

    __local float acc1_local[2][64];
    __local float acc2_local[2][64];
    acc1_local[get_local_id(0)][row_sz2_block] = acc1;
    acc2_local[get_local_id(0)][row_sz2_block] = acc2;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0)
    {
        acc1 = acc1_local[0][row_sz2_block] + acc1_local[1][row_sz2_block];
        acc2 = acc2_local[0][row_sz2_block] + acc2_local[1][row_sz2_block];
        out[out_qblock * 128 + row_sz2_block * 2] = acc1;
        out[out_qblock * 128 + row_sz2_block * 2 + 1] = acc2;
    }
}
