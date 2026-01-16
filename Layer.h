#pragma once

#include <concepts>
#include "Tensor.h"

template<typename T>
concept Layer = requires(
    T layer,
    const Tensor4D& input,
    const Tensor4D& grad_output,
    float learning_rate
) {
    { layer.forward(input) } -> std::same_as<Tensor4D>;
    { layer.backward(grad_output) } -> std::same_as<Tensor4D>;
    { layer.update_weights(learning_rate) } -> std::same_as<void>;
    { layer.zero_gradients() } -> std::same_as<void>;

    // metadata for debugging, could probs hide if needed for a release build
    { layer.name() } -> std::convertible_to<std::string>;
    { layer.parameter_count() } -> std::same_as<size_t>;
};
