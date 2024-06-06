// one day, but atm zig spirv support isn't good enough
export fn vec_add(a: [*]addrspace(.global) f32, b: [*]addrspace(.global) f32, c: [*]addrspace(.global) f32, n: i32) callconv(.Kernel) void {
    const i = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
