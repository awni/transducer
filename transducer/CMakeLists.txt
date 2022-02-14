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
  ${CMAKE_CURRENT_SOURCE_DIR}/transducer_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transducer_cpu.h
)

add_library(transducer ${TRANSDUCER_SOURCES})

if(OpenMP_CXX_FOUND)
  add_definitions(-D_OPENMP_)
  target_link_libraries(transducer PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CUDA_FOUND)
  add_definitions(-D_CUDA_)
  include_directories(${CUDA_INCLUDE_DIRS})
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>)
  target_sources(
    transducer
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/transducer_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/transducer_cuda.h
  )
endif()

if (TRANSDUCER_BUILD_TESTS)
  add_executable(test test.cpp)
  target_link_libraries(test PRIVATE transducer)
  add_executable(benchmark benchmark.cpp)
  target_link_libraries(benchmark PRIVATE transducer)
  if (CUDA_FOUND)
    add_executable(test_cuda test_cuda.cu)
    target_link_libraries(test_cuda PRIVATE transducer)
    add_executable(benchmark_cuda benchmark_cuda.cu)
    target_link_libraries(benchmark_cuda PRIVATE transducer)
  endif()
endif()

if (TRANSDUCER_BUILD_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings.")
  include(${CMAKE_MODULE_PATH}/Buildpybind11.cmake)
  include(${CMAKE_MODULE_PATH}/pybind11Tools.cmake)
  pybind11_add_module(_transducer _transducer.cpp)
  target_link_libraries(_transducer PRIVATE transducer)
  add_dependencies(_transducer pybind11)
endif()
