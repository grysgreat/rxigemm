# cublas_gemm_benchmark
add_executable(operator_matrix_test
    ${CMAKE_CURRENT_SOURCE_DIR}/operator_matrix_test.cu
)
add_executable(cublas_lib_test
    ${CMAKE_CURRENT_SOURCE_DIR}/cublas_lib_test.cu
)

target_link_libraries(operator_matrix_test -lcublas -lcurand)
target_link_libraries(cublas_lib_test -lcublas -lcurand)




include_directories(${SOURCE_DIR}/matrix)

target_link_libraries(operator_matrix_test
    matrix
)

target_link_libraries(cublas_lib_test
    matrix
)
