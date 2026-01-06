#include <unsupported/Eigen/CXX11/Tensor>
// #include <Eigen/Dense>
#include <iostream>

struct Tensor {
    Tensor() {
        std::cout << "tensor called" << std::endl;        
        tensor_ = Eigen::Tensor<int, 1>(len);
        for (size_t i{}; i < len; i++) {
            tensor_(i) = 21 + i;
        }
    }

    void print() {
        for (size_t i{}; i < len; i++) {
            std::cout << "Element at i: " << i << " is " << tensor_(i) << std::endl;
        }
    }

    size_t len{10};
    size_t dim{1};
    Eigen::Tensor<int, 1> tensor_;
};