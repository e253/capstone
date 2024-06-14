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
    const cl_h = b.dependency("cl_headers_upstream", .{});
    const cl_hpp = b.dependency("cl_hpp", .{});

    // build OCL ICD
    const ocl_icd_upstream = b.dependency("ocl_icd", .{});
    const opencl = b.addStaticLibrary(.{
        .name = "OpenCL",
        .target = target,
        .optimize = optimize,
    });
    const flags = [_][]const u8{
        "-DCL_TARGET_OPENCL_VERSION=300",
        "-DCL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES",
        "-DOPENCL_ICD_LOADER_VERSION_MAJOR=3",
        "-DOPENCL_ICD_LOADER_VERSION_MINOR=0",
        "-DOPENCL_ICD_LOADER_VERSION_REV=6",
    };
    opencl.addCSourceFiles(.{
        .root = ocl_icd_upstream.path("loader"),
        .files = &.{
            "icd.c",
            "icd_dispatch.c",
            "icd_dispatch_generated.c",
        },
        .flags = &flags,
    });
    if (target.result.os.tag == .windows) {
        opencl.addCSourceFiles(.{
            .root = ocl_icd_upstream.path("loader"),
            .files = &.{
                "windows/icd_windows.c",
                "windows/icd_windows_dxgk.c",
                "windows/icd_windows_envvars.c",
                "windows/icd_windows_hkr.c",
                "windows/icd_windows_apppackage.c",
            },
            .flags = &flags,
        });
        opencl.linkSystemLibrary("cfgmgr32");
        opencl.linkSystemLibrary("Ole32"); // runtimeobject
    } else if (target.result.os.tag == .linux) {
        opencl.addCSourceFiles(.{
            .root = ocl_icd_upstream.path("loader"),
            .files = &.{
                "linux/icd_linux.c",
                "linux/icd_linux_envvars.c",
            },
            .flags = &flags,
        });
        const icd_config_header = b.addConfigHeader(.{
            .style = .blank,
            .include_path = "icd_cmake_config.h",
        }, .{
            // we know this becuase musl libc linked by zig provides these functions.
            .HAVE_SECURE_GETENV = true,
            .HAVE___SECURE_GETENV = true,
        });
        opencl.addConfigHeader(icd_config_header);
    }
    opencl.addIncludePath(cl_h.path(""));
    opencl.addIncludePath(ocl_icd_upstream.path("loader"));
    opencl.linkLibC();
    b.installArtifact(opencl);

    // ===== collect cl_kernels into static library =====
    const cl_src = b.addStaticLibrary(.{
        .name = "clsrc",
        .root_source_file = .{ .path = "src/kernels/cl_src.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(cl_src);

    // ===== test_ref ======
    {
        const exe = b.addExecutable(.{
            .name = "test_ref",
            .target = target,
            .optimize = optimize,
        });
        exe.addCSourceFiles(.{
            .root = .{ .path = "." },
            .files = &.{
                "test/test_ref.cpp",
                "src/ref.cpp",
            },
            .flags = &.{
                "-Wall",
                "-Werror",
                "-std=c++17",
            },
        });
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
        exe.addCSourceFiles(.{
            .root = .{ .path = "." },
            .files = &.{
                "test/test_impl.cpp",
                "src/cpu.cpp",
                "src/thread.cpp",
                "src/ref.cpp",
            },
            .flags = &.{
                "-Wall",
                "-Werror",
                "-std=c++17",
                "-mavx512f",
                "-mavx512bw",
            },
        });
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
        exe.addCSourceFiles(.{
            .root = .{ .path = "." },
            .files = &.{
                "src/igpu.cpp",
                "src/ref.cpp",
                "test/test_igpu.cpp",
            },
            .flags = &.{
                "-Wall",
                "-Werror",
                "-std=c++17",
            },
        });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.addIncludePath(cl_h.path(""));
        exe.addIncludePath(cl_hpp.path("include"));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(gtest);
        exe.linkLibrary(opencl);
        exe.linkLibrary(cl_src);

        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("test_igpu", "Test IGPU Implementation");
        run_step.dependOn(&run_cmd.step);
    }

    // ===== bench igpu =====
    {
        const exe = b.addExecutable(.{
            .name = "bench_igpu",
            .target = target,
            .optimize = .ReleaseFast,
        });
        exe.addCSourceFiles(.{
            .root = .{ .path = "." },
            .files = &.{
                "bench/igpu.cpp",
                "src/igpu.cpp",
            },
            .flags = &.{
                "-Wall",
                "-Werror",
                "-std=c++17",
            },
        });
        exe.addIncludePath(.{ .path = "include" });
        exe.addIncludePath(gtest_upstream.path("googletest/include"));
        exe.addIncludePath(cl_h.path(""));
        exe.addIncludePath(cl_hpp.path("include"));
        exe.linkLibC();
        exe.linkLibCpp();
        exe.linkLibrary(gtest);
        exe.linkLibrary(opencl);
        exe.linkLibrary(cl_src);
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
        exe.addCSourceFiles(.{
            .root = .{
                .path = "bench",
            },
            .files = &.{
                "ggml.cpp",
            },
            .flags = &.{
                "-std=c++17",
            },
        });
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
        exe.addCSourceFiles(.{
            .root = .{
                .path = ".",
            },
            .files = &.{
                "src/cpu.cpp",
                "src/thread.cpp",
                "bench/cpu.cpp",
            },
            .flags = &.{
                "-std=c++17",
                "-mavx512f",
                "-O3",
            },
        });
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
