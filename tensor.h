#include <unsupported/Eigen/CXX11/Tensor>
// #include <Eigen/Dense>
#include <iostream>
#include <concepts>
#include <vector>
#include <array>
#include <type_traits>
#include <tuple>

// TODO:
// still debating whether it should be arithmetic
// doesn't make sense to have non-floats for cnn tbh
// but what if we want to just extrapolate into a
// general tensor lib which should support ints too
// will change module anyway when we implement CUDA
// template<typename T>
// concept TensorType = std::is_floating_point_v<T>;


// in: dims, out: tensor with dims defined
// from shape array unpack the vals into tuple
// construct tensor with tuple
// return tensor
template<size_t Dims>
auto pack_tensor(const std::array<size_t, Dims>& shape) {
    auto tuple = std::apply([](auto... args) {
        return std::make_tuple(args...);
    }, shape);

    auto tensor = std::apply([](auto... args) {
        return Eigen::Tensor<float, Dims>(args...);
    }, tuple);

    return tensor;
}

template<size_t Dims>
class Tensor {
private:
    Eigen::Tensor<float, Dims> tensor_;
    std::array<size_t, Dims> shape_;

public:
    // TODO: do we need move/copy/dtor ops? ro5 if we do end up w/one tho
    explicit Tensor(const std::array<size_t, Dims>& shape) : shape_(shape) {
        // fill out tensor with shape and vals? 
        tensor_ = pack_tensor<Dims>(shape);
    }

    template<typename... Indices>
    float& operator[](Indices... indices) {
        return tensor_(indices...);
    }
    // shape
    // idx op
    // fill
    // zero
    // data
    // random_normal
    Eigen::Tensor<float, Dims> dangerous_get_og_container() {
        return tensor_;
    }

};

using Tensor1D = Tensor<1>;
using Tensor2D = Tensor<2>;
using Tensor4D = Tensor<4>;