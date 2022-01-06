cmake_minimum_required(VERSION 3.8)
project(CUDA_MAT_MUL LANGUAGES CXX CUDA)

add_library(cudaMatMul cudaMatMul.cu cudaMatMul.h)
target_compile_features(cudaMatMul PUBLIC cxx_std_11)

add_executable(main_test main.cc)
target_link_libraries(main_test cudaMatMul)
