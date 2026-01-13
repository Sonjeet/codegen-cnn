#include <gtest/gtest.h>
#include <array>
#include "linalg.h"

TEST(LinAlgTest, MatrixMultiply) {
    linalg::Matrix A(2, 2), B(2, 2);
    A << 1, 2, 3, 4;
    B << 2, 0, 1, 3;
    
    linalg::Matrix result = linalg::multiply(A, B);
    EXPECT_FLOAT_EQ(result(0,0), 4.0f);
}

TEST(LinAlgTest, MatrixVectorMultiply) {
    linalg::Matrix A(2, 2);
    linalg::Vector x(2);
    A << 1, 2, 3, 4;
    x << 5, 6;
    
    linalg::Vector result = linalg::multiply(A, x);
    EXPECT_FLOAT_EQ(result(0), 17.0f);
}

TEST(LinAlgTest, HadamardProduct) {
    linalg::Vector a(2), b(2);
    a << 2, 3;
    b << 4, 5;
    
    linalg::Vector result = linalg::hadamard_product(a, b);
    EXPECT_FLOAT_EQ(result(0), 8.0f);
    EXPECT_FLOAT_EQ(result(1), 15.0f);
}

TEST(LinAlgTest, Sigmoid) {
    linalg::Vector x(1);
    x << 0.0f;
    
    linalg::Vector result = linalg::sigmoid(x);
    EXPECT_FLOAT_EQ(result(0), 0.5f);
}

TEST(LinAlgTest, SigmoidDerivative) {
    linalg::Vector sigmoid_out(1);
    sigmoid_out << 0.5f;
    
    linalg::Vector result = linalg::sigmoid_derivative(sigmoid_out);
    EXPECT_FLOAT_EQ(result(0), 0.25f);
}

TEST(LinAlgTest, Relu) {
    linalg::Vector x(3);
    x << -1.0f, 0.0f, 2.0f;
    
    linalg::Vector result = linalg::relu(x);
    EXPECT_FLOAT_EQ(result(0), 0.0f);
    EXPECT_FLOAT_EQ(result(1), 0.0f);
    EXPECT_FLOAT_EQ(result(2), 2.0f);
}

TEST(LinAlgTest, ReluDerivative) {
    linalg::Vector x(3);
    x << -1.0f, 0.0f, 2.0f;
    
    linalg::Vector result = linalg::relu_derivative(x);
    EXPECT_FLOAT_EQ(result(0), 0.0f);
    EXPECT_FLOAT_EQ(result(1), 0.0f);
    EXPECT_FLOAT_EQ(result(2), 1.0f);
}

TEST(LinAlgTest, TensorToMatrix) {
    std::array<size_t, 2> shape{2, 2};
    Tensor2D tensor(shape);
    
    tensor(0, 0) = 1.0f;
    tensor(0, 1) = 3.0f;
    tensor(1, 0) = 10.0f;
    tensor(1, 1) = 2.0f;

    linalg::Matrix matrix = linalg::tensor_to_matrix(tensor);
    EXPECT_FLOAT_EQ(matrix(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(matrix(0, 1), 3.0f);
    EXPECT_FLOAT_EQ(matrix(1, 0), 10.0f);
    EXPECT_FLOAT_EQ(matrix(1, 1), 2.0f);
}
