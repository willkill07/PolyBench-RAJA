cmake_minimum_required(VERSION 3.1)

SET(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/dist")
SET(RAJA_ENABLE_TESTS OFF CACHE BOOL "RAJA Tests")
SET(RAJA_ENABLE_NESTED ON CACHE BOOL "ForallN")
SET(RAJA_ENABLE_OPENMP ON CACHE BOOL "OpenMP")
add_subdirectory(RAJA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # Intel uses non-standard host-target optimizations :P
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHOST")
  # CMake doesn't understand Intel
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  # host-target optimization
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Local Headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# RAJA Headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/RAJA/include)
# RAJA_config
include_directories(${CMAKE_CURRENT_BINARY_DIR}/RAJA/include)

find_library(RT_LIBRARIES rt)

add_library(PolyBench src/PolyBenchKernel.cpp)

enable_testing()

macro(add_polybench_test test)
  add_executable(${test}
    src/${test}.cpp)

  target_link_libraries(${test} PolyBench RAJA rt)
  execute_process(COMMAND grep -i ${test} ${CMAKE_CURRENT_SOURCE_DIR}/common/polybench.spec COMMAND cut -f7 COMMAND head -n1 OUTPUT_VARIABLE testargs OUTPUT_STRIP_TRAILING_WHITESPACE)
  separate_arguments(testargs)
  add_test(NAME test_${test} COMMAND ${PROJECT_BINARY_DIR}/${test} ${testargs})

endmacro()

add_polybench_test(2mm)
add_polybench_test(3mm)
add_polybench_test(adi)
add_polybench_test(atax)
add_polybench_test(bicg)
add_polybench_test(cholesky)
add_polybench_test(correlation)
add_polybench_test(covariance)
add_polybench_test(deriche)
add_polybench_test(doitgen)
add_polybench_test(durbin)
add_polybench_test(fdtd-2d)
add_polybench_test(floyd-warshall)
add_polybench_test(gemm)
add_polybench_test(gemver)
add_polybench_test(gesummv)
add_polybench_test(gramschmidt)
add_polybench_test(heat-3d)
add_polybench_test(jacobi-1d)
add_polybench_test(jacobi-2d)
add_polybench_test(lu)
add_polybench_test(ludcmp)
add_polybench_test(mvt)
add_polybench_test(nussinov)
add_polybench_test(seidel-2d)
add_polybench_test(symm)
add_polybench_test(syr2k)
add_polybench_test(syrk)
add_polybench_test(trisolv)
add_polybench_test(trmm)
