#include <gtest/gtest.h>
#include <array>
#include "tensor.h"

TEST(TensorTest, InitTensor) {
    std::array<size_t, 1> arr{10};
    Tensor1D t(arr);

    EXPECT_EQ(t.size(), 10);
}

TEST(TensorTest, ConstructMultiDims) {
    std::array<size_t, 1> arr1{10};
    std::array<size_t, 2> arr2{2,3};
    std::array<size_t, 4> arr3{2, 3, 4, 5};

    Tensor1D t1(arr1);
    Tensor2D t2(arr2);
    Tensor4D t4(arr3);

    EXPECT_EQ(t1.shape()[0], 10);
    EXPECT_EQ(t2.shape()[0], 2);
    EXPECT_EQ(t2.shape()[1], 3);
    EXPECT_EQ(t4.shape()[0], 2);
    EXPECT_EQ(t4.shape()[1], 3);
    EXPECT_EQ(t4.shape()[2], 4);
    EXPECT_EQ(t4.shape()[3], 5);
}

TEST(TensorTest, FillTensor) {
    std::array<size_t, 1> arr{10};
    Tensor1D t(arr);
    
    for (size_t i{}; i < 10; i++) {
        EXPECT_EQ(t[i], 0);
    }
    
    t.fill(42);
    for (size_t i{}; i < 10; i++) {
        EXPECT_EQ(t[i], 42);
    }
}