cmake_minimum_required(VERSION 3.18)
project(vdbhnsw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Tune for the build host's instruction set (AVX2/FMA or AVX-512) so the VCL
# distance kernels in distances_simd.hpp pick Vec8f/Vec16f accordingly.
add_compile_options(-march=native)

find_package(pybind11 CONFIG REQUIRED)

set(VCL_INCLUDE_DIR third_party/vcl)

pybind11_add_module(_vdbhnsw src/bindings.cpp)
target_include_directories(_vdbhnsw PRIVATE include ${VCL_INCLUDE_DIR})

# Land the extension module at the wheel root so `import _vdbhnsw` works as a
# top-level module alongside the `vdbbench` package.
install(TARGETS _vdbhnsw LIBRARY DESTINATION .)

# Standalone tests (not part of the Python build): build with
#   cmake -S native/hnsw -B native/hnsw/build-test -DBUILD_TESTS=ON
#   cmake --build native/hnsw/build-test
option(BUILD_TESTS "Build standalone C++ tests" OFF)
if(BUILD_TESTS)
  add_executable(test_hnsw tests/test_hnsw.cpp)
  target_include_directories(test_hnsw PRIVATE include ${VCL_INCLUDE_DIR})
  target_compile_options(test_hnsw PRIVATE -O3)

  add_executable(test_distances tests/test_distances.cpp)
  target_include_directories(test_distances PRIVATE include ${VCL_INCLUDE_DIR})
  target_compile_options(test_distances PRIVATE -O3)
endif()
