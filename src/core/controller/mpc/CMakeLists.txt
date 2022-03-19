cmake_minimum_required(VERSION 3.0.2)
project(mpc)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

find_package(Eigen3 3.3 REQUIRED)
find_package(PythonLibs REQUIRED)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   INCLUDE_DIRS include
#  LIBRARIES mpc
   CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

include_directories(/usr/include)
link_directories(/usr/lib)

add_library(mpc src/MPC.cpp)
target_link_libraries(mpc ipopt ${catkin_LIBRARIES})

add_executable(mpc_test src/main.cpp)
# target_link_libraries(mpc_test mpc ${catkin_LIBRARIES})
target_link_libraries(mpc_test mpc ipopt ${catkin_LIBRARIES} ${PYTHON_LIBRARIES})
