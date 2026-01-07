find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

enable_testing()
add_executable(cnn_tests
    TestTensor.cpp) 

target_link_libraries(
    cnn_tests
    Eigen3::Eigen
    GTest::gtest_main)

target_include_directories(cnn_tests PRIVATE ${CMAKE_SOURCE_DIR})

include(GoogleTest)

gtest_discover_tests(cnn_tests)
