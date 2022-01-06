cmake_minimum_required(VERSION 3.8)
project(cuda_mat_mul LANGUAGES CXX CUDA)

find_package(catkin REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES cuda_mat_mul
    )

include_directories(
    #SYSTEM
    include 
    ${catkin_INCLUDE_DIRS}
)

#add_library(cuda_mat_mul src/cudaMatMul.cu src/cudaMatMul.h)
add_library(cuda_mat_mul src/cudaMatMul.cu)
#target_compile_features(cudaMatMul PUBLIC cxx_std_11)
target_link_libraries(cuda_mat_mul ${catkin_LIBRARIES})



add_executable(cudamatmulalpha src/main.cc)
target_link_libraries(cudamatmulalpha cuda_mat_mul)

add_executable(cublasMatMul src/cublasMatMul.cu)
target_link_libraries(cublasMatMul
   ${catkin_LIBRARIES}
   #   -lcuda
   -lcublas
   #-lcurand
)
