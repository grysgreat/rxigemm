# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/may_sparse/xigemm_cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/may_sparse/xigemm_cuda/build

# Include any dependencies generated for this target.
include matrix/test/CMakeFiles/operator_matrix_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include matrix/test/CMakeFiles/operator_matrix_test.dir/compiler_depend.make

# Include the progress variables for this target.
include matrix/test/CMakeFiles/operator_matrix_test.dir/progress.make

# Include the compile flags for this target's objects.
include matrix/test/CMakeFiles/operator_matrix_test.dir/flags.make

matrix/test/CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o: matrix/test/CMakeFiles/operator_matrix_test.dir/flags.make
matrix/test/CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o: ../matrix/test/operator_matrix_test.cu
matrix/test/CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o: matrix/test/CMakeFiles/operator_matrix_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/may_sparse/xigemm_cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object matrix/test/CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o"
	cd /root/may_sparse/xigemm_cuda/build/matrix/test && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT matrix/test/CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o -MF CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o.d -x cu -c /root/may_sparse/xigemm_cuda/matrix/test/operator_matrix_test.cu -o CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o

matrix/test/CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

matrix/test/CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target operator_matrix_test
operator_matrix_test_OBJECTS = \
"CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o"

# External object files for target operator_matrix_test
operator_matrix_test_EXTERNAL_OBJECTS = \
"/root/may_sparse/xigemm_cuda/build/matrix/CMakeFiles/matrix.dir/gen_matrix.cu.o" \
"/root/may_sparse/xigemm_cuda/build/matrix/CMakeFiles/matrix.dir/print_matrix.cu.o" \
"/root/may_sparse/xigemm_cuda/build/matrix/CMakeFiles/matrix.dir/operator_matrix.cu.o"

matrix/test/operator_matrix_test: matrix/test/CMakeFiles/operator_matrix_test.dir/operator_matrix_test.cu.o
matrix/test/operator_matrix_test: matrix/CMakeFiles/matrix.dir/gen_matrix.cu.o
matrix/test/operator_matrix_test: matrix/CMakeFiles/matrix.dir/print_matrix.cu.o
matrix/test/operator_matrix_test: matrix/CMakeFiles/matrix.dir/operator_matrix.cu.o
matrix/test/operator_matrix_test: matrix/test/CMakeFiles/operator_matrix_test.dir/build.make
matrix/test/operator_matrix_test: matrix/test/CMakeFiles/operator_matrix_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/may_sparse/xigemm_cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable operator_matrix_test"
	cd /root/may_sparse/xigemm_cuda/build/matrix/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/operator_matrix_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
matrix/test/CMakeFiles/operator_matrix_test.dir/build: matrix/test/operator_matrix_test
.PHONY : matrix/test/CMakeFiles/operator_matrix_test.dir/build

matrix/test/CMakeFiles/operator_matrix_test.dir/clean:
	cd /root/may_sparse/xigemm_cuda/build/matrix/test && $(CMAKE_COMMAND) -P CMakeFiles/operator_matrix_test.dir/cmake_clean.cmake
.PHONY : matrix/test/CMakeFiles/operator_matrix_test.dir/clean

matrix/test/CMakeFiles/operator_matrix_test.dir/depend:
	cd /root/may_sparse/xigemm_cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/may_sparse/xigemm_cuda /root/may_sparse/xigemm_cuda/matrix/test /root/may_sparse/xigemm_cuda/build /root/may_sparse/xigemm_cuda/build/matrix/test /root/may_sparse/xigemm_cuda/build/matrix/test/CMakeFiles/operator_matrix_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : matrix/test/CMakeFiles/operator_matrix_test.dir/depend

