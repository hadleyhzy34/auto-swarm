cmake_minimum_required(VERSION 3.0.2)
project(second LANGUAGES CXX CUDA)

## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    cuda_mat_mul
    #cudaMatMul
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES second
    CATKIN_DEPENDS cuda_mat_mul
#  DEPENDS system_lib
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)

add_executable(secondalpha src/main.cc)
target_link_libraries(secondalpha 
    ${catkin_LIBRARIES}
)
