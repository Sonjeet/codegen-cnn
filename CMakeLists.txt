cmake_minimum_required(VERSION 3.20.0)
project(Codegen-CNN)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3p libs
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# target_include_directories(${EIGEN_INCLUDE_DIR})

add_executable(cnn
    main.cpp
    tensor.h)

target_link_libraries(
    cnn
    Eigen3::Eigen
)

enable_testing()
add_executable(cnn_tests
    HelloTest.cpp)


target_link_libraries(
    cnn_tests
    GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(cnn_tests)