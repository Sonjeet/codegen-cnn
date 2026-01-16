#include <gtest/gtest.h>
#include <array>
#include "Tensor.h"

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
        EXPECT_EQ(t(i), 0);
    }
    
    t.fill(42);
    for (size_t i{}; i < 10; i++) {
        EXPECT_EQ(t(i), 42);
    }
}

TEST(TensorTest, ZeroOperation) {
    std::array<size_t, 2> shape{3, 3};
    Tensor2D t(shape);
    
    t.fill(99.0f);
    t.zero();
    
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(t(i, j), 0.0f);
        }
    }
}

TEST(TensorTest, ElementAccess4D) {
    std::array<size_t, 4> shape{2, 3, 4, 5};
    Tensor4D t(shape);
    
    t(1, 2, 3, 4) = 42.0f;
    EXPECT_FLOAT_EQ(t(1, 2, 3, 4), 42.0f);
    EXPECT_FLOAT_EQ(t(0, 0, 0, 0), 0.0f);
}

TEST(TensorTest, SizeCalculation) {
    std::array<size_t, 4> shape{2, 3, 4, 5};
    Tensor4D t(shape);
    
    EXPECT_EQ(t.size(), 2 * 3 * 4 * 5);
}

TEST(TensorTest, DataPointerAccess) {
    std::array<size_t, 1> shape{5};
    Tensor1D t(shape);
    
    float* data = t.data();
    data[0] = 1.0f;
    data[4] = 5.0f;
    
    EXPECT_FLOAT_EQ(t(0), 1.0f);
    EXPECT_FLOAT_EQ(t(4), 5.0f);
}

TEST(TensorTest, RandomNormalDistribution) {
    std::array<size_t, 1> shape{1000};
    Tensor1D t(shape);
    
    t.random_normal(0.0f, 1.0f);
    
    float sum = 0.0f;
    for (size_t i = 0; i < t.size(); i++) {
        sum += t(i);
    }
    float mean = sum / t.size();
    
    EXPECT_NEAR(mean, 0.0f, 0.2f);
}

TEST(TensorTest, ConstDataAccess) {
    std::array<size_t, 1> shape{3};
    Tensor1D t(shape);
    t.fill(7.0f);
    
    const Tensor1D& const_ref = t;
    const float* data = const_ref.data();
    
    EXPECT_FLOAT_EQ(data[0], 7.0f);
    EXPECT_FLOAT_EQ(data[2], 7.0f);
}