## EGEMV On Intel SoCs

This repository houses my undergraduate thesis: *EGEMV On Intel SoCs*.

The full report is in the repo in the `/report` folder and hosted through github pages: [https://e253.github.io/capstone/report/Steere_Ethan_B__EGEMV_On_Intel_SoCs.pdf](https://e253.github.io/capstone/report/Steere_Ethan_B__EGEMV_On_Intel_SoCs.pdf)

I've continued to make pushes after graduation in May and hope to continue making improvements when I can.

Here's a few things I'd like to try:

- [ ] Separate OpenCL kernels into separate files.
- [ ] Compile `OpenCL-C` to SPIR-V at build-time instead of runtime.
  - This is possible with [KhronosGroup/SPIRV-LLVM-Translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator). SPIRV-LLVM-Translator is highly inconvient to build and use. Maybe it needs a `build.zig`.
  - Alternatively, Zig with `callconv(.Kernel)` can generate PTX and SPIRV kernels. Maybe that's a better solution. 
  - I hate building `OpenCL-C` at runtime. It's slow, could fail randomly, and most of all feels janky. It's a non-solution that I would feel nervous shipping it. Embedding SPIR-V is a solution similar to `nvcc`. 
- [ ] Use [Level Zero](https://github.com/oneapi-src/level-zero) to execute OpenCL/SPIR-V. This [article](https://jjfumero.github.io/posts/2021/09/introduction-to-level-zero/) shows how Level-Zero can analyze the kernel and suggest work group sizes. There is also other features for controlling temperature. I had problems enqueing dependent kernels ... hopefully ze can do that better.
