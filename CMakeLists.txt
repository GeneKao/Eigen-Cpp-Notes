cmake_minimum_required(VERSION 3.17)
project(eigen_cpp_notes)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(EIGEN_DIR "./external/eigen")
if(NOT EIGEN_DIR)
    find_package (Eigen3 3.3 REQUIRED NO_MODULE)
endif()
include_directories(${EIGEN_DIR})

set(src main.cpp)
add_executable(eigen_cpp_notes main.cpp)