#include <CL/cl.h>
#include <CL/cl_function_types.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef _WIN32

#include <libloaderapi.h>

typedef HMODULE libocl_t;

#define LOAD_OCL() LoadLibrary(TEXT("opencl.dll"))
#define FREE_LIBRARY(lib) FreeLibrary(lib)
#define GET_SYM(name, func_type) ((func_type)(GetProcAddress(libocl, name)))
#define CHECK_DL_ERROR() (fputs("TODO error handling on windows.", stderr))

#else

#include <dlfcn.h>

typedef void* libocl_t;

#define LOAD_OCL() dlopen("libOpenCL.so", RTLD_LAZY)
#define FREE_LIBRARY(lib) dlclose(lib)
#define GET_SYM(name, func_type) ((func_type)(dlsym(libocl, name)))
#define CHECK_DL_ERROR()          \
    {                             \
        char* error = dlerror();  \
        if (error != NULL) {      \
            fputs(error, stderr); \
            exit(1);              \
        }                         \
    }

#endif // _WIN32

static libocl_t libocl;
static bool initialized = false;

/* rocl Functions */

// Loads `OpenCL.dll` or `libopencl.so`
// It will reload the library if it was already loaded
// Returns 0 on success, 1 on failure
bool rocl_init()
{
    libocl = LOAD_OCL();
    if (libocl == NULL) {
        return false;
    } else {
        initialized = true;
        return true;
    }
}

// Frees `OpenCL.dll` / `libopencl.so`
// Safe even if `rocl_init` failed
void rocl_deinit()
{
    if (libocl != NULL) {
        FREE_LIBRARY(libocl);
    }
}

// Returns string description for OpenCL error code
// Thanks Selmar on StackOverflow: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clErrorToString(cl_int err)
{
    switch (err) {
    // run-time and JIT compiler errors
    case 0:
        return "CL_SUCCESS";
    case -1:
        return "CL_DEVICE_NOT_FOUND";
    case -2:
        return "CL_DEVICE_NOT_AVAILABLE";
    case -3:
        return "CL_COMPILER_NOT_AVAILABLE";
    case -4:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:
        return "CL_OUT_OF_RESOURCES";
    case -6:
        return "CL_OUT_OF_HOST_MEMORY";
    case -7:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8:
        return "CL_MEM_COPY_OVERLAP";
    case -9:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case -10:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11:
        return "CL_BUILD_PROGRAM_FAILURE";
    case -12:
        return "CL_MAP_FAILURE";
    case -13:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case -16:
        return "CL_LINKER_NOT_AVAILABLE";
    case -17:
        return "CL_LINK_PROGRAM_FAILURE";
    case -18:
        return "CL_DEVICE_PARTITION_FAILED";
    case -19:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30:
        return "CL_INVALID_VALUE";
    case -31:
        return "CL_INVALID_DEVICE_TYPE";
    case -32:
        return "CL_INVALID_PLATFORM";
    case -33:
        return "CL_INVALID_DEVICE";
    case -34:
        return "CL_INVALID_CONTEXT";
    case -35:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case -36:
        return "CL_INVALID_COMMAND_QUEUE";
    case -37:
        return "CL_INVALID_HOST_PTR";
    case -38:
        return "CL_INVALID_MEM_OBJECT";
    case -39:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40:
        return "CL_INVALID_IMAGE_SIZE";
    case -41:
        return "CL_INVALID_SAMPLER";
    case -42:
        return "CL_INVALID_BINARY";
    case -43:
        return "CL_INVALID_BUILD_OPTIONS";
    case -44:
        return "CL_INVALID_PROGRAM";
    case -45:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46:
        return "CL_INVALID_KERNEL_NAME";
    case -47:
        return "CL_INVALID_KERNEL_DEFINITION";
    case -48:
        return "CL_INVALID_KERNEL";
    case -49:
        return "CL_INVALID_ARG_INDEX";
    case -50:
        return "CL_INVALID_ARG_VALUE";
    case -51:
        return "CL_INVALID_ARG_SIZE";
    case -52:
        return "CL_INVALID_KERNEL_ARGS";
    case -53:
        return "CL_INVALID_WORK_DIMENSION";
    case -54:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case -55:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case -56:
        return "CL_INVALID_GLOBAL_OFFSET";
    case -57:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case -58:
        return "CL_INVALID_EVENT";
    case -59:
        return "CL_INVALID_OPERATION";
    case -60:
        return "CL_INVALID_GL_OBJECT";
    case -61:
        return "CL_INVALID_BUFFER_SIZE";
    case -62:
        return "CL_INVALID_MIP_LEVEL";
    case -63:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64:
        return "CL_INVALID_PROPERTY";
    case -65:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66:
        return "CL_INVALID_COMPILER_OPTIONS";
    case -67:
        return "CL_INVALID_LINKER_OPTIONS";
    case -68:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000:
        return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001:
        return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002:
        return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003:
        return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004:
        return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005:
        return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default:
        return "Unknown OpenCL error";
    }
}

/* CL Dispatch Functions */

typedef struct rocl_dispatch_table_s {
    clGetPlatformIDs_fn _clGetPlatformIDs;
    clGetPlatformInfo_fn _clGetPlatformInfo;
    clGetDeviceIDs_fn _clGetDeviceIDs;
    clGetDeviceInfo_fn _clGetDeviceInfo;
    // clCreateContext_fn _clCreateContext;
} rocl_dispatch_table_t;

static rocl_dispatch_table_t dispatch_table = { 0 };

extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    if (dispatch_table._clGetPlatformIDs == NULL) {
        dispatch_table._clGetPlatformIDs = GET_SYM("clGetPlatformIDs", clGetPlatformIDs_fn);
        CHECK_DL_ERROR();
    }

    return dispatch_table._clGetPlatformIDs(num_entries, platforms, num_platforms);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    if (dispatch_table._clGetPlatformInfo == NULL) {
        dispatch_table._clGetPlatformInfo = GET_SYM("clGetPlatformInfo", clGetPlatformInfo_fn);
        CHECK_DL_ERROR();
    }

    return dispatch_table._clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices) CL_API_SUFFIX__VERSION_1_0
{
    if (dispatch_table._clGetDeviceIDs == NULL) {
        dispatch_table._clGetDeviceIDs = GET_SYM("clGetDeviceIDs", clGetDeviceIDs_fn);
        CHECK_DL_ERROR();
    }

    return dispatch_table._clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    if (dispatch_table._clGetDeviceInfo == NULL) {
        dispatch_table._clGetDeviceInfo = GET_SYM("clGetDeviceInfo", clGetDeviceInfo_fn);
        CHECK_DL_ERROR();
    }

    return dispatch_table._clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
}