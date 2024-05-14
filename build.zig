const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build Google Test
    const gtest_upstream = b.dependency("gtest_upstream", .{});
    const gtest = b.addStaticLibrary(.{
        .name = "gtest",
        .target = target,
        .optimize = optimize,
    });
    gtest.linkLibC();
    gtest.linkLibCpp();
    gtest.addCSourceFiles(.{ .root = gtest_upstream.path("googletest"), .files = &.{"src/gtest-all.cc"}, .flags = &.{"-std=c++14"} });
    gtest.addIncludePath(gtest_upstream.path("googletest/include"));
    gtest.addIncludePath(gtest_upstream.path("googletest"));
    b.installArtifact(gtest);

    const argparse = b.dependency("argparse", .{});
    const cl_headers_upstream = b.dependency("cl_headers_upstream", .{});

    // build rocl
    const rocl = b.addStaticLibrary(.{
        .name = "rocl",
        .target = target,
        .optimize = optimize,
    });
    rocl.addCSourceFiles(.{ .files = &.{"src/rocl.c"}, .flags = &.{"-DCL_TARGET_OPENCL_VERSION=300"} });
    rocl.linkLibC();
    rocl.addIncludePath(cl_headers_upstream.path(""));
    b.installArtifact(rocl);

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
        }, .flags = &.{ "-Wall", "-Werror", "-std=c++17" } });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(gtest);
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
            "src/ref.cpp",
        }, .flags = &.{ "-Wall", "-Werror", "-std=c++17", "-mavx512f", "-mavx512bw" } });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(gtest);
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

    // ===== test igpu =====
    {
        const exe = b.addExecutable(.{
            .name = "test_igpu",
            .target = target,
            .optimize = optimize,
        });
        exe.addCSourceFiles(.{ .root = .{ .path = "src" }, .files = &.{
            "igpu.cpp",
            "ref.cpp",
        }, .flags = &.{ "-Wall", "-Werror", "-std=c++17" } });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.addIncludePath(cl_headers_upstream.path(""));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(gtest);
        if (target.result.os.tag == .windows) {
            exe.addLibraryPath(.{ .path = "C:\\Windows\\System32" });
            exe.linkSystemLibrary2("opencl", .{ .preferred_link_mode = .dynamic });
        } else {
            exe.linkSystemLibrary("OpenCL");
        }
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("test_igpu", "Test IGPU Implementation");
        run_step.dependOn(&run_cmd.step);
    }

    // test_rocl
    {
        const exe = b.addExecutable(.{
            .name = "test_rocl",
            .target = target,
            .optimize = optimize,
        });
        exe.addCSourceFiles(.{ .root = .{
            .path = "test",
        }, .files = &.{
            "test_rocl.cpp",
        }, .flags = &.{
            "-std=c++17",
        } });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.addIncludePath(cl_headers_upstream.path(""));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(gtest);
        exe.linkLibrary(rocl);
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("test_rocl", "Test rocl.h");
        run_step.dependOn(&run_cmd.step);
    }

    // ===== bench igpu =====
    {
        const exe = b.addExecutable(.{
            .name = "bench_igpu",
            .target = target,
            .optimize = .ReleaseFast,
        });
        exe.addCSourceFiles(.{ .root = .{ .path = "src" }, .files = &.{
            "igpu.cpp",
        }, .flags = &.{ "-DBENCH", "-Wall", "-Werror", "-std=c++17" } });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.addIncludePath(cl_headers_upstream.path(""));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(gtest);
        if (target.result.os.tag == .windows) {
            exe.addLibraryPath(.{ .path = "C:\\Windows\\System32" });
            exe.linkSystemLibrary2("opencl", .{ .preferred_link_mode = .dynamic });
        } else {
            exe.linkSystemLibrary("OpenCL");
        }
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("bench_igpu", "Test IGPU Implementation");
        run_step.dependOn(&run_cmd.step);
    }

    // ===== bench ggml =====
    {
        const exe = b.addExecutable(.{
            .name = "bench_ggml",
            .target = target,
            .optimize = .ReleaseFast,
        });
        exe.addCSourceFiles(.{ .root = .{
            .path = "bench",
        }, .files = &.{
            "ggml.cpp",
        }, .flags = &.{
            "-std=c++17",
        } });
        exe.addIncludePath(.{ .path = "include" });
        exe.linkLibC();
        exe.linkLibCpp();
        exe.addIncludePath(argparse.path("include"));
        if (target.result.os.tag == .windows) {
            exe.addLibraryPath(.{ .path = "C:/Program Files (x86)/ggml/lib" });
            exe.addIncludePath(.{ .path = "C:/Program Files (x86)/ggml/include" });
        }
        exe.linkSystemLibrary("ggml");
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("bench_ggml", "Benchmark GGML");
        run_step.dependOn(&run_cmd.step);
    }

    // ===== bench cpu =====
    {
        const exe = b.addExecutable(.{
            .name = "bench_cpu",
            .target = target,
            .optimize = .ReleaseFast,
        });
        exe.addCSourceFiles(.{ .root = .{
            .path = ".",
        }, .files = &.{
            "src/cpu.cpp",
            "src/thread.cpp",
            "bench/cpu.cpp",
        }, .flags = &.{
            "-std=c++17",
            "-mavx512f",
            "-O3",
        } });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(argparse.path("include"));
        exe.linkLibC();
        exe.linkLibCpp();
        if (target.result.os.tag == .linux) {
            exe.linkSystemLibrary("pthread");
        }
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("bench_cpu", "Benchmark CPU");
        run_step.dependOn(&run_cmd.step);
    }
}
