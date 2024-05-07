const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build Google Test
    const gtest_upstream = b.dependency("gtest_upstream", .{});
    const libgtest = b.addStaticLibrary(.{
        .name = "libgtest",
        .target = target,
        .optimize = optimize,
    });
    libgtest.linkLibC();
    libgtest.linkLibCpp();
    libgtest.addCSourceFiles(.{ .root = gtest_upstream.path("googletest"), .files = &.{"src/gtest-all.cc"}, .flags = &.{"-std=c++14"} });
    libgtest.addIncludePath(gtest_upstream.path("googletest/include"));
    libgtest.addIncludePath(gtest_upstream.path("googletest"));
    b.installArtifact(libgtest);
    const cl_headers_upstream = b.dependency("cl_headers_upstream", .{});
    _ = cl_headers_upstream;

    // ===== test_ref ======
    {
        const exe = b.addExecutable(.{
            .name = "test_ref",
            .target = target,
            .optimize = optimize,
        });
        exe.addCSourceFiles(.{ .root = .{ .path = "." }, .files = &.{
            "test/test_ref.cpp",
            "src/ref.cpp",
        }, .flags = &.{ "-Wall", "-Werror", "-std=c++14" } });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(libgtest);
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("test_ref", "Test Reference Implementation");
        run_step.dependOn(&run_cmd.step);
    }

    // ===== test_cpu ======
    {
        const exe = b.addExecutable(.{
            .name = "test_cpu",
            .target = target,
            .optimize = optimize,
        });
        exe.addCSourceFiles(.{ .root = .{ .path = "." }, .files = &.{
            "test/test_impl.cpp",
            "src/cpu.cpp",
            "src/thread.cpp",
        }, .flags = &.{ "-Wall", "-Werror", "-std=c++14", "-mavx512f", "-mavx512bw" } });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(libgtest);
        if (target.result.os.tag == .linux) {
            exe.linkSystemLibrary("pthread");
        }
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("test_cpu", "Test CPU Implementation");
        run_step.dependOn(&run_cmd.step);
    }
}
