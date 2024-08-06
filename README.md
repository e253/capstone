## EGEMV On Intel SoCs

This repository houses my undergraduate thesis: *EGEMV On Intel SoCs*.

The full report is in the repo in the `/report` folder: [Steere_Ethan_B__EGEMV_On_Intel_SoCs.pdf](https://raw.githubusercontent.com/e253/capstone/main/report/Steere_Ethan_B__EGEMV_On_Intel_SoCs.pdf)

I've continued to make pushes after graduation in May and hope to continue making improvements when I can.

Here's a few things I'd like to try:

- [x] Separate OpenCL kernels into separate files. (6/14)
- [ ] Compile `OpenCL-C` to SPIR-V at build-time instead of runtime.
  - This is possible with [KhronosGroup/SPIRV-LLVM-Translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator). SPIRV-LLVM-Translator is highly inconvient to build and use. Maybe it needs a `build.zig`.
  - Alternatively, Zig with `callconv(.Kernel)` can generate PTX and SPIRV kernels. Maybe that's a better solution. 
  - I feel uncomfortable building `OpenCL-C` at runtime. It's slow and feels janky. Embedding SPIR-V would provide an experience similar to `nvcc`. 
- [ ] Use [Level Zero](https://github.com/oneapi-src/level-zero) to execute OpenCL/SPIR-V. This [article](https://jjfumero.github.io/posts/2021/09/introduction-to-level-zero/) shows that Level-Zero, unlike OpenCL, can analyze the kernel and suggest work group sizes.
