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

    // test_ref.cpp
    const test_exe = b.addExecutable(.{
        .name = "test_ref",
        .target = target,
        .optimize = optimize,
    });
    test_exe.addCSourceFiles(.{ .root = .{ .path = "." }, .files = &.{
        "test/test_ref.cpp",
        "src/ref.cpp",
    }, .flags = &.{ "-Wall", "-Werror", "-std=c++14" } });
    test_exe.addIncludePath(.{ .path = "include" });
    test_exe.addIncludePath(gtest_upstream.path("googletest/include"));
    test_exe.linkLibC();
    test_exe.linkLibCpp();
    test_exe.linkLibrary(libgtest);
    b.installArtifact(test_exe);

    const run_cmd = b.addRunArtifact(test_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("test_ref", "Test Reference Implementation");
    run_step.dependOn(&run_cmd.step);
}
