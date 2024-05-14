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

int main(int argc, char** argv)
{
    bool ocl_loaded = rocl_init();
    if (!ocl_loaded) {
        cout << "OCL was not found on the system, tests cannot run." << endl;
        cout << "if `clinfo` finds things, then there is a problem" << endl;
        return 0;
    }
    testing::InitGoogleTest(&argc, argv);
    int test_ret = RUN_ALL_TESTS();

    rocl_deinit();

    return test_ret;
}
