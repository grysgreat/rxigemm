# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cublas_gemm_benchmark LANGUAGES CXX CUDA)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -arch=sm_75")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -arch=sm_75")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -arch=sm_75")
endif()

add_subdirectory(matrix)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RCXIGEMM_BIN_NAME rcxigemm)


include_directories(${SOURCE_DIR})
include_directories(${SOURCE_DIR}/matrix)




# cublas_gemm_benchmark
add_executable(${RCXIGEMM_BIN_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/run_matrix.cu
)

include_directories(${SOURCE_DIR}/matrix)

target_link_libraries(${RCXIGEMM_BIN_NAME}
    matrix
)



target_link_libraries(${RCXIGEMM_BIN_NAME} -lcublas -lcurand)



