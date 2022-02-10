cmake_minimum_required(VERSION 3.10)

project(transducer LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(CUDA 9.0)
find_package(OpenMP)

option(TRANSDUCER_BUILD_TESTS "Build tests" ON)
option(TRANSDUCER_BUILD_PYTHON_BINDINGS "Build python bindings" OFF)
option(TRANSDUCER_BUILD_CUDA "Build cuda backend" ${CUDA_FOUND})

set(
  TRANSDUCER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/transducer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transducer.h
)

add_library(transducer ${TRANSDUCER_SOURCES})

if(OpenMP_CXX_FOUND)
  #    add_definitions(-D_CUDA_)
    target_link_libraries(transducer PUBLIC OpenMP::OpenMP_CXX)
endif()

if (TRANSDUCER_BUILD_TESTS)
    add_executable(test test.cpp)
    target_link_libraries(test PRIVATE transducer)
#    add_executable(benchmarks benchmarks.cpp)
#    target_link_libraries(benchmarks PRIVATE transducer)
endif()

if (TRANSDUCER_BUILD_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings.")
  include(${CMAKE_MODULE_PATH}/Buildpybind11.cmake)
  include(${CMAKE_MODULE_PATH}/pybind11Tools.cmake)
  pybind11_add_module(_transducer _transducer.cpp)
  target_link_libraries(_transducer PRIVATE transducer)
  add_dependencies(_transducer pybind11)
endif()
