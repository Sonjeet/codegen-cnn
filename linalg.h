#pragma once
#include <Eigen/Dense>
#include "Tensor.h"

/**
 * Lot of the eigen algs are documented under ArrayBase
 * https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1ArrayBase.html
 * 
 * TODO: optimisation? we're returning copies in a few of these methods. we could
 * just pass in a non-const ref/ptr as an arg that should be filled
 * therefore no copy? present before/after profiling on blog
 */
namespace linalg {
    using Matrix = Eigen::MatrixXf;
    using Vector = Eigen::VectorXf;
    
    // Matrix operations
    Matrix multiply(const Matrix& A, const Matrix& B) {
        return A * B;
    }
    
    Vector multiply(const Matrix& A, const Vector& x) {
        return A * x;
    }
    
    // Element-wise operations
    Matrix hadamard_product(const Matrix& A, const Matrix& B) {
        return A.cwiseProduct(B);
    }
    
    Vector hadamard_product(const Vector& a, const Vector& b) {
        return a.cwiseProduct(b);
    }
    
    // Activation functions
    Vector sigmoid(const Vector& x) {
        return (1.0f / (1.0f + (-x.array()).exp())).matrix();
    }
    
    Vector sigmoid_derivative(const Vector& sigmoid_output) {
        return hadamard_product(sigmoid_output, 
                               (1.0f - sigmoid_output.array()).matrix());
    }
    
    Vector relu(const Vector& x) {
        return x.cwiseMax(0.0f);
    }
    
    Vector relu_derivative(const Vector& x) {
        return (x.array() > 0.0f)
            .cast<float>() // eigen cast, not cpp
            .matrix();
    }
    
    // Tensor to Eigen conversion helpers
    Matrix tensor_to_matrix(const Tensor2D& tensor) {
        return Eigen::Map<const Matrix>(tensor.data(), 
                                       tensor.shape()[0], 
                                       tensor.shape()[1]);
    }
    
    void matrix_to_tensor(const Matrix& matrix, Tensor2D& tensor) {
        std::copy(matrix.data(), matrix.data() + matrix.size(), tensor.data());
    }
};
