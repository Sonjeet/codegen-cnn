#include <iostream>
#include <array>
#include "tensor.h"

int main(int argc, char* argv[]) {
    std::cout << "hey" << std::endl;
    // std::array<size_t, 1> arr{1};
    std::array<size_t, 1> arr{10};
    Tensor1D t(arr);

    for(size_t i{}; i < tensor.size(); i++) {
        std::cout << t[i];
    }    

    return 0;
}