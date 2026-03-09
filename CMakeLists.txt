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

# Print configuration
message(STATUS "")
message(STATUS "========== Adding-Doubling Configuration ==========")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3: fetched via FetchContent (3.4.0)")
message(STATUS "===================================================")
message(STATUS "")
