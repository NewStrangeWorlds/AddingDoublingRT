cmake_minimum_required(VERSION 3.15)
project(adding_doubling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /DEIGEN_NO_DEBUG")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_NO_DEBUG -march=native")
endif()

# Dependencies: Eigen3 via FetchContent
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Library
add_library(adding_doubling
    src/solver.cpp
    src/planck.cpp
    src/quadrature.cpp
    src/phase_matrix.cpp
)
target_link_libraries(adding_doubling PUBLIC Eigen3::Eigen)
target_include_directories(adding_doubling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Example executable
add_executable(ad_example src/example.cpp)
target_link_libraries(ad_example adding_doubling)

# Test suite
enable_testing()
add_executable(ad_tests tests/main.cpp tests/test_ad_solver.cpp)
target_link_libraries(ad_tests adding_doubling)
target_include_directories(ad_tests PRIVATE tests)
add_test(NAME ad_tests COMMAND ad_tests)

# CUDA support (optional)
include(CheckLanguage)
check_language(CUDA)

option(ADRT_ENABLE_CUDA "Enable CUDA support for batched solver" OFF)

if(ADRT_ENABLE_CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "ADRT_ENABLE_CUDA is ON but no CUDA compiler found")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    add_library(adding_doubling_cuda
        src_cuda/cuda_solver.cu
    )
    target_link_libraries(adding_doubling_cuda PUBLIC adding_doubling)
    target_include_directories(adding_doubling_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src_cuda)

    target_compile_options(adding_doubling_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            -lineinfo
        >
    )

    # CUDA test suite
    add_executable(ad_cuda_tests
        tests/test_cuda_solver.cu
    )
    target_link_libraries(ad_cuda_tests adding_doubling adding_doubling_cuda)
    target_include_directories(ad_cuda_tests PRIVATE tests)
    add_test(NAME ad_cuda_tests COMMAND ad_cuda_tests)

    # CUDA benchmark
    add_executable(ad_cuda_benchmark
        tests/benchmark_cuda.cu
    )
    target_link_libraries(ad_cuda_benchmark adding_doubling adding_doubling_cuda)
    target_include_directories(ad_cuda_benchmark PRIVATE tests)
endif()

# Print configuration
message(STATUS "")
message(STATUS "========== Adding-Doubling Configuration ==========")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3: fetched via FetchContent (3.4.0)")
if(ADRT_ENABLE_CUDA)
    message(STATUS "CUDA: enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "CUDA: disabled (set -DADRT_ENABLE_CUDA=ON to enable)")
endif()
message(STATUS "===================================================")
message(STATUS "")
