#include <iostream>
#include <array>
#include "Tensor.h"

// running a simple test... nothing special
int main(int argc, char* argv[]) {
    std::array<size_t, 1> arr{10};
    Tensor1D tensor(arr);
    tensor(3) = 33333;

    for(size_t i{}; i < tensor.size(); i++) {
        std::cout << tensor(i);
    }

    return 0;
}