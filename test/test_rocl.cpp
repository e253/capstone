#include "CL/cl.h"
#include "CL/cl_function_types.h"
#include "rocl/rocl.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

TEST(QueryAPI, GetPlatformIDs)
{
    cl_uint num_platforms;
    cl_int ret = clGetPlatformIDs(0, NULL, &num_platforms);
    ASSERT_EQ(ret, CL_SUCCESS);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);
    ASSERT_EQ(ret, CL_SUCCESS);
    free(platforms);
}

TEST(QueryAPI, clGetPlatformInfo)
{
    cl_uint num_platforms;
    cl_int ret = clGetPlatformIDs(0, NULL, &num_platforms);
    ASSERT_EQ(ret, CL_SUCCESS);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);
    ASSERT_EQ(ret, CL_SUCCESS);

    for (cl_uint i = 0; i < num_platforms; i++) {
        size_t param_value_size;
        ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &param_value_size);
        ASSERT_EQ(ret, CL_SUCCESS);
        char* param_value = (char*)malloc(param_value_size);
        ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, param_value_size, param_value, NULL);
        ASSERT_EQ(ret, CL_SUCCESS);
        free(param_value);
    }

    free(platforms);
}

TEST(QueryAPI, GetDeviceIDs)
{
    cl_uint num_platforms;
    cl_int ret = clGetPlatformIDs(0, NULL, &num_platforms);
    ASSERT_EQ(ret, CL_SUCCESS);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);
    ASSERT_EQ(ret, CL_SUCCESS);

    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        ASSERT_EQ(ret, CL_SUCCESS);
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        ASSERT_EQ(ret, CL_SUCCESS);
        free(devices);
    }

    free(platforms);
}

TEST(QueryAPI, GetDeviceInfo)
{
    cl_uint num_platforms;
    cl_int ret = clGetPlatformIDs(0, NULL, &num_platforms);
    ASSERT_EQ(ret, CL_SUCCESS);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);
    ASSERT_EQ(ret, CL_SUCCESS);

    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        ASSERT_EQ(ret, CL_SUCCESS);
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        ASSERT_EQ(ret, CL_SUCCESS);

        for (cl_uint j = 0; j < num_devices; j++) {
            size_t param_value_size;
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &param_value_size);
            ASSERT_EQ(ret, CL_SUCCESS);
            char* param_value = (char*)malloc(param_value_size);
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, param_value_size, param_value, NULL);
            ASSERT_EQ(ret, CL_SUCCESS);
            free(param_value);
        }

        free(devices);
    }

    free(platforms);
}

TEST(ConextAPI, CreateAndReleaseContext)
{
    cl_uint num_platforms;
    cl_int ret = clGetPlatformIDs(0, NULL, &num_platforms);
    ASSERT_EQ(ret, CL_SUCCESS);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(num_platforms, platforms, NULL);
    ASSERT_EQ(ret, CL_SUCCESS);

    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        ASSERT_EQ(ret, CL_SUCCESS);
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        ASSERT_EQ(ret, CL_SUCCESS);

        cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &ret);
        ASSERT_EQ(ret, CL_SUCCESS);
        ret = clReleaseContext(context);
        ASSERT_EQ(ret, CL_SUCCESS);

        free(devices);
    }

    free(platforms);
}

int main(int argc, char** argv)
{
    // bool ocl_loaded = rocl_init();
    //  if (!ocl_loaded) {
    //      cout << "OCL was not found on the system, tests cannot run." << endl;
    //      cout << "if `clinfo` finds things, then there is a problem" << endl;
    //      return 0;
    //  }
    testing::InitGoogleTest(&argc, argv);
    int test_ret = RUN_ALL_TESTS();

    // rocl_deinit();

    return test_ret;
}
