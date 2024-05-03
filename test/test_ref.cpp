#include "capstone/capstone.hpp"
#include "gtest/gtest.h"
#include <iostream>

int mul_by_2(int a)
{
    return a * 2;
}

TEST(mul_by_2, Positive)
{
    EXPECT_EQ(4, mul_by_2(2));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
