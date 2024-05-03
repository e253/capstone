const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // const lib = b.addStaticLibrary() -- SOON

    const test_exe = b.addExecutable(.{
        .name = "test_ref",
        .root_source_file = .{ .path = "./test/test_ref.cpp" },
        .target = target,
        .optimize = optimize,
    });
    test_exe.addCSourceFiles(&.{"./src/ref.cpp"}, &.{"-std=c++17"});
    test_exe.addIncludePath(.{ .path = "./include" });
    test_exe.linkLibC();
    test_exe.linkLibCpp();

    b.installArtifact(test_exe);
    const run_cmd = b.addRunArtifact(test_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("test_ref", "Test Reference Implementation");
    run_step.dependOn(&run_cmd.step);
}
