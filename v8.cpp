#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

void asm_ukernel_v8(
    uint8_t* w, // Weight, offset from the global pointer
    uint64_t w_cs, // Col stride for weights
    float* scales, // Weight scales, offset from the global pointer
    uint64_t scales_cs, // Col stride for scales
    uint8_t* zeros, // Weight zeros, offset from the global pointer
    uint64_t zeros_cs, // Col stride for zeros
    float* in, // input, offset from the global pointer
    float* out, // out, offset from the global pointer
    uint64_t cols)
{

    asm volatile(
        "movq      %0,      %%rsi    \n\t" // w
        "movq      %1,      %%rdi    \n\t" // w_cs
        "movq      %2,      %%rax    \n\t" // scales
        "movq      %3,      %%rbx    \n\t" // scales_cs
        "leaq      (,%%rbx,4),%%rbx  \n\t" // scales_cs *= 4
        "movq      %4,      %%r8     \n\t" // zeros
        "movq      %5,      %%r9     \n\t" // zeros_cs
        "movq      %6,      %%r10    \n\t" // in
        "movq      %7,      %%r11    \n\t" // out
        "movq      %8,      %%r12    \n\t" // cols
        "decq      %%r12             \n\t" // cols -= 1

        // Set load mask (k1)
        "movl     $0x00FF,  %%r13d   \n\t"
        "kmovw    %%r13d,   %%k1     \n\t"

        // Set and mask (xmm0)
        "movl $0x0F0F0F0F,  %%r13d   \n"
        "movd      %%r13d,  %%xmm0   \n"
        "pshufd $0,%%xmm0,  %%xmm0   \n"

        // Initialize Scales
        "vmovups (%%rax),      %%zmm9  \n\t"
        "vmovups 4*16(%%rax),  %%zmm10 \n\t"
        "vmovups 4*16*2(%%rax),%%zmm11 \n\t"
        "vmovups 4*16*3(%%rax),%%zmm12 \n\t"
        "vmovups 4*16*4(%%rax),%%zmm13 \n\t"
        "vmovups 4*16*5(%%rax),%%zmm14 \n\t"
        "vmovups 4*16*6(%%rax),%%zmm15 \n\t"
        "vmovups 4*16*7(%%rax),%%zmm16 \n\t"
        "leaq (%%rax,%%rbx),%%rax      \n\t"

        // Initialize Zeros.
        "vmovdqu8 (%%r8),  %%xmm1%{%%k1%}%{z%} \n\t"
        "vmovdqu8 8(%%r8), %%xmm2%{%%k1%}%{z%} \n\t"
        "vmovdqu8 16(%%r8),%%xmm3%{%%k1%}%{z%} \n\t"
        "vmovdqu8 24(%%r8),%%xmm4%{%%k1%}%{z%} \n\t"
        "vmovdqu8 32(%%r8),%%xmm5%{%%k1%}%{z%} \n\t"
        "vmovdqu8 40(%%r8),%%xmm6%{%%k1%}%{z%} \n\t"
        "vmovdqu8 48(%%r8),%%xmm7%{%%k1%}%{z%} \n\t"
        "vmovdqu8 56(%%r8),%%xmm8%{%%k1%}%{z%} \n\t"
        "leaq (%%r8,%%r9),%%r8 \n\t" // zeros += zeros_cs

        "vmovdqu64 %%xmm1,%%xmm29 \n\t"
        "vmovdqu64 %%xmm2,%%xmm30 \n\t"
        "vmovdqu64 %%xmm3,%%xmm31 \n\t"
        "vmovdqu64 %%xmm4,%%xmm25 \n\t"

        "vpandd %%xmm29,%%xmm0,%%xmm29 \n\t"
        "vpandd %%xmm30,%%xmm0,%%xmm30 \n\t"
        "vpandd %%xmm31,%%xmm0,%%xmm31 \n\t"
        "vpandd %%xmm25,%%xmm0,%%xmm25 \n\t"

        "vpsrld $4,%%xmm1,%%xmm1 \n\t"
        "vpsrld $4,%%xmm2,%%xmm2 \n\t"
        "vpsrld $4,%%xmm3,%%xmm3 \n\t"
        "vpsrld $4,%%xmm4,%%xmm4 \n\t"

        "vpandd %%xmm1,%%xmm0,%%xmm1 \n\t"
        "vpandd %%xmm2,%%xmm0,%%xmm2 \n\t"
        "vpandd %%xmm3,%%xmm0,%%xmm3 \n\t"
        "vpandd %%xmm4,%%xmm0,%%xmm4 \n\t"

        "vpunpcklbw %%xmm1,%%xmm29,%%xmm1 \n\t"
        "vpunpcklbw %%xmm2,%%xmm30,%%xmm2 \n\t"
        "vpunpcklbw %%xmm3,%%xmm31,%%xmm3 \n\t"
        "vpunpcklbw %%xmm4,%%xmm25,%%xmm4 \n\t"

        "vmovdqu64 %%xmm5,%%xmm29 \n\t"
        "vmovdqu64 %%xmm6,%%xmm30 \n\t"
        "vmovdqu64 %%xmm7,%%xmm31 \n\t"
        "vmovdqu64 %%xmm8,%%xmm25 \n\t"

        "vpandd %%xmm29,%%xmm0,%%xmm29 \n\t"
        "vpandd %%xmm30,%%xmm0,%%xmm30 \n\t"
        "vpandd %%xmm31,%%xmm0,%%xmm31 \n\t"
        "vpandd %%xmm25,%%xmm0,%%xmm25 \n\t"

        "vpsrld $4,%%xmm5,%%xmm5 \n\t"
        "vpsrld $4,%%xmm6,%%xmm6 \n\t"
        "vpsrld $4,%%xmm7,%%xmm7 \n\t"
        "vpsrld $4,%%xmm8,%%xmm8 \n\t"

        "vpandd %%xmm5,%%xmm0,%%xmm5 \n\t"
        "vpandd %%xmm6,%%xmm0,%%xmm6 \n\t"
        "vpandd %%xmm7,%%xmm0,%%xmm7 \n\t"
        "vpandd %%xmm8,%%xmm0,%%xmm8 \n\t"

        "vpunpcklbw %%xmm5,%%xmm29,%%xmm5 \n\t"
        "vpunpcklbw %%xmm6,%%xmm30,%%xmm6 \n\t"
        "vpunpcklbw %%xmm7,%%xmm31,%%xmm7 \n\t"
        "vpunpcklbw %%xmm8,%%xmm25,%%xmm8 \n\t"

        // Set accumulators to zero (zmm9-zmm16)
        "vpxorq %%zmm17,%%zmm17,%%zmm17 \n\t"
        "vpxorq %%zmm18,%%zmm18,%%zmm18 \n\t"
        "vpxorq %%zmm19,%%zmm19,%%zmm19 \n\t"
        "vpxorq %%zmm20,%%zmm20,%%zmm20 \n\t"
        "vpxorq %%zmm21,%%zmm21,%%zmm21 \n\t"
        "vpxorq %%zmm22,%%zmm22,%%zmm22 \n\t"
        "vpxorq %%zmm23,%%zmm23,%%zmm23 \n\t"
        "vpxorq %%zmm24,%%zmm24,%%zmm24 \n\t"

        // Main Loop
        "xorq %%rcx,%%rcx \n\t"
        ".MAINLOOP%=:     \n\t"

        // Load Input
        "vbroadcastss (%%r10),%%zmm29 \n\t"
        "addq $4,%%r10 \n\t" // in += 1 (4 bytes)

        // ======= 1 + 2 =======
        "vmovdqu8 (%%rsi),%%xmm25%{%%k1%}%{z%}  \n\t" // 1: load 16 weights to lower 64 bits
        "vmovdqu8 8(%%rsi),%%xmm26%{%%k1%}%{z%} \n\t" // 2

        "vmovdqa64 %%xmm25,%%xmm30 \n\t" // 1: weights --> tmp
        "vmovdqa64 %%xmm26,%%xmm31 \n\t" // 2

        "vpandq %%xmm30,%%xmm0,%%xmm30 \n\t" // 1: tmp &= 0x0F
        "vpandq %%xmm31,%%xmm0,%%xmm31 \n\t" // 2

        "vpsrld $4,%%xmm25,%%xmm25 \n\t" // 1: weights >> 4
        "vpsrld $4,%%xmm26,%%xmm26 \n\t" // 2

        "vpandq %%xmm25,%%xmm0,%%xmm25 \n\t" // 1: weights &= 0x0F
        "vpandq %%xmm26,%%xmm0,%%xmm26 \n\t" // 2

        "vpunpcklbw %%xmm25,%%xmm30,%%xmm25 \n\t" // 1: weights = weights[0],tmp[1]...
        "vpunpcklbw %%xmm26,%%xmm31,%%xmm26 \n\t" // 2

        "vpsubb %%xmm1,%%xmm25,%%xmm25 \n\t" // 1: weights -= zeros
        "vpsubb %%xmm2,%%xmm26,%%xmm26 \n\t" // 2

        "vpmovsxbd %%xmm25,%%zmm25 \n\t" // 1: weights = (int)weights
        "vpmovsxbd %%xmm26,%%zmm26 \n\t" // 2

        "vcvtdq2ps %%zmm25,%%zmm25 \n\t" // 1: weights = (float)weights
        "vcvtdq2ps %%zmm26,%%zmm26 \n\t" // 2

        "vmulps %%zmm9,%%zmm25,%%zmm25  \n\t" // 1: weights *= scale
        "vmulps %%zmm10,%%zmm26,%%zmm26 \n\t" // 2

        "vfmadd231ps %%zmm29,%%zmm25,%%zmm17 \n\t" // 1: acc += weights * input
        "vfmadd231ps %%zmm29,%%zmm26,%%zmm18 \n\t" // 2

        // ======= 3 + 4 =======
        "vmovdqu8 16(%%rsi),%%xmm27%{%%k1%}%{z%} \n\t" // 3: load 16 weights to lower 64 bits
        "vmovdqu8 24(%%rsi),%%xmm28%{%%k1%}%{z%} \n\t" // 4

        "vmovdqa64 %%xmm27,%%xmm30 \n\t" // 3: weights --> tmp
        "vmovdqa64 %%xmm28,%%xmm31 \n\t" // 4

        "vpandq %%xmm30,%%xmm0,%%xmm30 \n\t" // 3: tmp &= 0x0F
        "vpandq %%xmm31,%%xmm0,%%xmm31 \n\t" // 4

        "vpsrld $4,%%xmm27,%%xmm27 \n\t" // 3: weights >> 4
        "vpsrld $4,%%xmm28,%%xmm28 \n\t" // 4

        "vpandq %%xmm27,%%xmm0,%%xmm27 \n\t" // 3: weights &= 0x0F
        "vpandq %%xmm28,%%xmm0,%%xmm28 \n\t" // 4

        "vpunpcklbw %%xmm27,%%xmm30,%%xmm27 \n\t" // 3: weights = weights[0],tmp[1]...
        "vpunpcklbw %%xmm28,%%xmm31,%%xmm28 \n\t" // 4

        "vpsubb %%xmm3,%%xmm27,%%xmm27 \n\t" // 3: weights -= zeros
        "vpsubb %%xmm4,%%xmm28,%%xmm28 \n\t" // 4

        "vpmovsxbd %%xmm27,%%zmm27 \n\t" // 3: weights = (int)weights
        "vpmovsxbd %%xmm28,%%zmm28 \n\t" // 4

        "vcvtdq2ps %%zmm27,%%zmm27 \n\t" // 3: weights = (float)weights
        "vcvtdq2ps %%zmm28,%%zmm28 \n\t" // 4

        "vmulps %%zmm11,%%zmm27,%%zmm27 \n\t" // 3: weights *= scale
        "vmulps %%zmm12,%%zmm28,%%zmm28 \n\t" // 4

        "vfmadd231ps %%zmm29,%%zmm27,%%zmm19 \n\t" // 3: acc += weights * input
        "vfmadd231ps %%zmm29,%%zmm28,%%zmm20 \n\t" // 4

        // ======= 5 + 6 =======
        "vmovdqu8 32(%%rsi),%%xmm25%{%%k1%}%{z%} \n\t" // 5
        "vmovdqu8 40(%%rsi),%%xmm26%{%%k1%}%{z%} \n\t" // 6

        "vmovdqa64 %%xmm25,%%xmm30 \n\t" // 5
        "vmovdqa64 %%xmm26,%%xmm31 \n\t" // 6

        "vpandq %%xmm30,%%xmm0,%%xmm30 \n\t" // 5
        "vpandq %%xmm31,%%xmm0,%%xmm31 \n\t" // 6

        "vpsrld $4,%%xmm25,%%xmm25 \n\t" // 5
        "vpsrld $4,%%xmm26,%%xmm26 \n\t" // 6

        "vpandq %%xmm25,%%xmm0,%%xmm25 \n\t" // 5
        "vpandq %%xmm26,%%xmm0,%%xmm26 \n\t" // 6

        "vpunpcklbw %%xmm25,%%xmm30,%%xmm25 \n\t" // 5
        "vpunpcklbw %%xmm26,%%xmm31,%%xmm26 \n\t" // 6

        "vpsubb %%xmm5,%%xmm25,%%xmm25 \n\t" // 5
        "vpsubb %%xmm6,%%xmm26,%%xmm26 \n\t" // 6

        "vpmovsxbd %%xmm25,%%zmm25 \n\t" // 5
        "vpmovsxbd %%xmm26,%%zmm26 \n\t" // 6

        "vcvtdq2ps %%zmm25,%%zmm25 \n\t" // 5
        "vcvtdq2ps %%zmm26,%%zmm26 \n\t" // 6

        "vmulps %%zmm13,%%zmm25,%%zmm25 \n\t" // 5
        "vmulps %%zmm14,%%zmm26,%%zmm26 \n\t" // 6

        "vfmadd231ps %%zmm29,%%zmm25,%%zmm21 \n\t" // 5
        "vfmadd231ps %%zmm29,%%zmm26,%%zmm22 \n\t" // 6

        // ======= 7 + 8 =======
        "vmovdqu8 48(%%rsi),%%xmm27%{%%k1%}%{z%} \n\t" // 7: load 16 weights to lower 64 bits
        "vmovdqu8 56(%%rsi),%%xmm28%{%%k1%}%{z%} \n\t" // 8

        "leaq (%%rsi,%%rdi),%%rsi \n\t" // weight += weights_cs (32 bytes)

        "vmovdqa64 %%xmm27,%%xmm30 \n\t" // 7: weights --> tmp
        "vmovdqa64 %%xmm28,%%xmm31 \n\t" // 8

        "vpandq %%xmm30,%%xmm0,%%xmm30 \n\t" // 7: tmp &= 0x0F
        "vpandq %%xmm31,%%xmm0,%%xmm31 \n\t" // 8

        "vpsrld $4,%%xmm27,%%xmm27 \n\t" // 7: weights >> 4
        "vpsrld $4,%%xmm28,%%xmm28 \n\t" // 8

        "vpandq %%xmm27,%%xmm0,%%xmm27 \n\t" // 7: weights &= 0x0F
        "vpandq %%xmm28,%%xmm0,%%xmm28 \n\t" // 8

        "vpunpcklbw %%xmm27,%%xmm30,%%xmm27 \n\t" // 7: weights = weights[0],tmp[1]...
        "vpunpcklbw %%xmm28,%%xmm31,%%xmm28 \n\t" // 8

        "vpsubb %%xmm7,%%xmm27,%%xmm27 \n\t" // 7: weights -= zeros
        "vpsubb %%xmm8,%%xmm28,%%xmm28 \n\t" // 8

        "vpmovsxbd %%xmm27,%%zmm27 \n\t" // 7: weights = (int)weights
        "vpmovsxbd %%xmm28,%%zmm28 \n\t" // 8

        "vcvtdq2ps %%zmm27,%%zmm27 \n\t" // 7: weights = (float)weights
        "vcvtdq2ps %%zmm28,%%zmm28 \n\t" // 8

        "vmulps %%zmm15,%%zmm27,%%zmm27 \n\t" // 7: weights *= scale
        "vmulps %%zmm16,%%zmm28,%%zmm28 \n\t" // 8

        "vfmadd231ps %%zmm29,%%zmm27,%%zmm23 \n\t" // 7: acc += weights * input
        "vfmadd231ps %%zmm29,%%zmm28,%%zmm24 \n\t" // 8

        // Maybe Roll Over Zeros and Scales?
        "testq     $0,      %%rcx \n\t"
        "je       .LOOPEPILOQUE%= \n\t"
        "movq      %%rcx,   %%r15 \n\t"
        "and       $127,    %%r15 \n\t"
        "testq     $0,      %%r15 \n\t"
        "jne      .LOOPEPILOQUE%= \n\t"

        // Scales
        "vmovups (%%rax),%%zmm9        \n\t"
        "vmovups 4*16(%%rax),%%zmm10   \n\t"
        "vmovups 4*16*2(%%rax),%%zmm11 \n\t"
        "vmovups 4*16*3(%%rax),%%zmm12 \n\t"
        "vmovups 4*16*4(%%rax),%%zmm13 \n\t"
        "vmovups 4*16*5(%%rax),%%zmm14 \n\t"
        "vmovups 4*16*6(%%rax),%%zmm15 \n\t"
        "vmovups 4*16*7(%%rax),%%zmm16 \n\t"
        "leaq (%%rax,%%rbx),%%rax       \n\t"

        // Zeros
        "vmovdqu8 (%%r8),%%xmm1%{%%k1%}%{z%}   \n\t"
        "vmovdqu8 8(%%r8),%%xmm2%{%%k1%}%{z%}  \n\t"
        "vmovdqu8 16(%%r8),%%xmm3%{%%k1%}%{z%} \n\t"
        "vmovdqu8 24(%%r8),%%xmm4%{%%k1%}%{z%} \n\t"
        "vmovdqu8 32(%%r8),%%xmm5%{%%k1%}%{z%} \n\t"
        "vmovdqu8 40(%%r8),%%xmm6%{%%k1%}%{z%} \n\t"
        "vmovdqu8 48(%%r8),%%xmm7%{%%k1%}%{z%} \n\t"
        "vmovdqu8 56(%%r8),%%xmm8%{%%k1%}%{z%} \n\t"
        "leaq (%%r8,%%r9),%%r8 \n\t"

        "vmovdqu64 %%xmm1,%%xmm29 \n\t"
        "vmovdqu64 %%xmm2,%%xmm30 \n\t"
        "vmovdqu64 %%xmm3,%%xmm31 \n\t"
        "vmovdqu64 %%xmm4,%%xmm25 \n\t"

        "vpandd %%xmm29,%%xmm0,%%xmm29 \n\t"
        "vpandd %%xmm30,%%xmm0,%%xmm30 \n\t"
        "vpandd %%xmm31,%%xmm0,%%xmm31 \n\t"
        "vpandd %%xmm25,%%xmm0,%%xmm25 \n\t"

        "vpsrld $4,%%xmm1,%%xmm1 \n\t"
        "vpsrld $4,%%xmm2,%%xmm2 \n\t"
        "vpsrld $4,%%xmm3,%%xmm3 \n\t"
        "vpsrld $4,%%xmm4,%%xmm4 \n\t"

        "vpandd %%xmm1,%%xmm0,%%xmm1 \n\t"
        "vpandd %%xmm2,%%xmm0,%%xmm2 \n\t"
        "vpandd %%xmm3,%%xmm0,%%xmm3 \n\t"
        "vpandd %%xmm4,%%xmm0,%%xmm4 \n\t"

        "vpunpcklbw %%xmm1,%%xmm29,%%xmm1 \n\t"
        "vpunpcklbw %%xmm2,%%xmm30,%%xmm2 \n\t"
        "vpunpcklbw %%xmm3,%%xmm31,%%xmm3 \n\t"
        "vpunpcklbw %%xmm4,%%xmm25,%%xmm4 \n\t"

        "vmovdqu64 %%xmm5,%%xmm29 \n\t"
        "vmovdqu64 %%xmm6,%%xmm30 \n\t"
        "vmovdqu64 %%xmm7,%%xmm31 \n\t"
        "vmovdqu64 %%xmm8,%%xmm25 \n\t"

        "vpandd %%xmm29,%%xmm0,%%xmm29 \n\t"
        "vpandd %%xmm30,%%xmm0,%%xmm30 \n\t"
        "vpandd %%xmm31,%%xmm0,%%xmm31 \n\t"
        "vpandd %%xmm25,%%xmm0,%%xmm25 \n\t"

        "vpsrld $4,%%xmm5,%%xmm5 \n\t"
        "vpsrld $4,%%xmm6,%%xmm6 \n\t"
        "vpsrld $4,%%xmm7,%%xmm7 \n\t"
        "vpsrld $4,%%xmm8,%%xmm8 \n\t"

        "vpandd %%xmm5,%%xmm0,%%xmm5 \n\t"
        "vpandd %%xmm6,%%xmm0,%%xmm6 \n\t"
        "vpandd %%xmm7,%%xmm0,%%xmm7 \n\t"
        "vpandd %%xmm8,%%xmm0,%%xmm8 \n\t"

        "vpunpcklbw %%xmm5,%%xmm29,%%xmm5 \n\t"
        "vpunpcklbw %%xmm6,%%xmm30,%%xmm6 \n\t"
        "vpunpcklbw %%xmm7,%%xmm31,%%xmm7 \n\t"
        "vpunpcklbw %%xmm8,%%xmm25,%%xmm8 \n\t"

        ".LOOPEPILOQUE%=:         \n\t"
        "incq     %%rcx           \n\t"
        "testq    %%r12,   %%rcx  \n\t"
        "jne      .MAINLOOP%=     \n\t"

        // Store Outputs
        "vmovups %%zmm17,      (%%r11) \n\t"
        "vmovups %%zmm18,  4*16(%%r11) \n\t"
        "vmovups %%zmm19,4*16*2(%%r11) \n\t"
        "vmovups %%zmm20,4*16*3(%%r11) \n\t"
        "vmovups %%zmm21,4*16*4(%%r11) \n\t"
        "vmovups %%zmm22,4*16*5(%%r11) \n\t"
        "vmovups %%zmm23,4*16*6(%%r11) \n\t"
        "vmovups %%zmm24,4*16*7(%%r11) \n\t"

        :
        : "m"(w),
        "m"(w_cs),
        "m"(scales),
        "m"(scales_cs),
        "m"(zeros),
        "m"(zeros_cs),
        "m"(in),
        "m"(out),
        "m"(cols)
        : "cc", "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8",
        "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory",
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8",
        "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "xmm16",
        "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22", "xmm23", "xmm24",
        "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31",
        "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8",
        "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16",
        "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24",
        "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
        "k1", "k2");
}

#define QBLOCK_SIZE 128

void test()
{
    int m = 128;
    int n = 512;

    uint8_t* Weights = (uint8_t*)_aligned_malloc(m * n / 2, 64);
    for (int i = 0; i < m * n / 2; i++)
        Weights[i] = 0x55;

    float* W_Scales = (float*)_aligned_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE; i++)
        W_Scales[i] = 2.0f;

    uint8_t* W_Zeros = (uint8_t*)_aligned_malloc(m * n / QBLOCK_SIZE / 2, 64);
    for (int i = 0; i < m * n / QBLOCK_SIZE / 2; i++)
        W_Zeros[i] = 0x11;

    float* Input = (float*)_aligned_malloc(n * sizeof(float), 64);
    for (int i = 0; i < n; i++)
        Input[i] = 2.0f;

    float* Output = (float*)_aligned_malloc(m * sizeof(float), 64);
    memset(Output, 0, m * sizeof(float));

    asm_ukernel_v8(
        Weights, m / 2,
        W_Scales, m,
        W_Zeros, m / 2,
        Input,
        Output,
        512);

    bool passed = true;
    for (int i = 0; i < m; i++) {
        if (Output[i] != 8192.0f) {
            cout << "Output[" << i << "] = " << Output[i] << endl;
            passed = false;
        }
    }
    if (!passed) {
        cout << endl;
        cout << "Tested Failed" << endl;
        exit(0);
    } else {
        cout << "Tested Passed" << endl;
    }
    std::cout.flush();

    _aligned_free(Weights);
    _aligned_free(W_Scales);
    _aligned_free(W_Zeros);
    _aligned_free(Input);
    _aligned_free(Output);
}

// column major indexing helper
#define CM(ptr, row, col, col_stride) ((ptr) + (col * (col_stride)) + (row))

void egemv(
    uint8_t* w,
    float* s,
    uint8_t* z,
    float* in,
    float* out,
    int m, int n)
{

    assert(m % 128 == 0 && "Row size must be divisble by 128");
    assert(n % QBLOCK_SIZE == 0 && "Col size must be divisble by 128");

    auto process_128_rows_512_cols = [&](uint8_t* w, float* s, uint8_t* z,
                                         float* in, float* out,
                                         int start_row, int end_row) {
        const int n_col_blocks = n / 512;

        for (int j = start_row; j < end_row; j += 128) {
            for (int col_block = 0; col_block < n_col_blocks; col_block++) {
                int i = col_block * 512;
                asm_ukernel_v8(
                    CM(w, j, i / 2, m / 2), m / 2,
                    CM(s, j, i / QBLOCK_SIZE, m), m,
                    CM(z, j, i / QBLOCK_SIZE / 2, m / 2), m / 2,
                    in + i, out + j,
                    512);
            }
        }
    };

    size_t n_threads = 4;
    vector<thread> threads(n_threads);

    int rows_per_thread = m / n_threads;
    assert(rows_per_thread % 128 == 0 && "Thread row blocks size must be divisible by 128");
    int start_row = 0;
    int end_row;
    for (int thread_id = 0; thread_id < n_threads; thread_id++) {
        end_row = start_row + rows_per_thread;
        threads[thread_id] = thread(process_128_rows_512_cols, w, s, z, in, out, start_row, end_row);
        start_row += rows_per_thread;
    }
    for (auto& t : threads) {
        t.join();
    }
    threads.clear();
}

void bench()
{
    uint64_t m = 4096;
    uint64_t n = 14336;

    uint8_t* w = (uint8_t*)_aligned_malloc(m * n / 2, 64);
    float* s = (float*)_aligned_malloc(m * n / QBLOCK_SIZE * sizeof(float), 64);
    uint8_t* z = (uint8_t*)_aligned_malloc(m * n / QBLOCK_SIZE / 2, 64);
    float* in = (float*)_aligned_malloc(n * sizeof(float), 64);
    float* out = (float*)_aligned_malloc(m * sizeof(float), 64);

    const int NIT = 500;

    auto start = chrono::high_resolution_clock::now();
    for (int it = 0; it < NIT; it++) {
        egemv(w, s, z, in, out, m, n);
    }
    auto end = chrono::high_resolution_clock::now();

    double sec = chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    cout << "total: " << sec << " (s)" << endl;
    cout << "ms/it: " << sec * 1000 / NIT << " (ms)" << endl;

    uint64_t flops_processed = m * n * 2 * NIT;
    double flops_sec = flops_processed / sec;
    double gflops = flops_sec / (1e9);
    cout << "GFLOPS: " << gflops << endl;

    _aligned_free(w);
    _aligned_free(s);
    _aligned_free(z);
    _aligned_free(in);
    _aligned_free(out);
}

int main()
{
    test();
    bench();
}