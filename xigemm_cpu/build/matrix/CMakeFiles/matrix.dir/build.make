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
CMAKE_SOURCE_DIR = /root/may_sparse/xigemm_cpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/may_sparse/xigemm_cpu/build

# Include any dependencies generated for this target.
include matrix/CMakeFiles/matrix.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include matrix/CMakeFiles/matrix.dir/compiler_depend.make

# Include the progress variables for this target.
include matrix/CMakeFiles/matrix.dir/progress.make

# Include the compile flags for this target's objects.
include matrix/CMakeFiles/matrix.dir/flags.make

matrix/CMakeFiles/matrix.dir/blas_mkl.cpp.o: matrix/CMakeFiles/matrix.dir/flags.make
matrix/CMakeFiles/matrix.dir/blas_mkl.cpp.o: ../matrix/blas_mkl.cpp
matrix/CMakeFiles/matrix.dir/blas_mkl.cpp.o: matrix/CMakeFiles/matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object matrix/CMakeFiles/matrix.dir/blas_mkl.cpp.o"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT matrix/CMakeFiles/matrix.dir/blas_mkl.cpp.o -MF CMakeFiles/matrix.dir/blas_mkl.cpp.o.d -o CMakeFiles/matrix.dir/blas_mkl.cpp.o -c /root/may_sparse/xigemm_cpu/matrix/blas_mkl.cpp

matrix/CMakeFiles/matrix.dir/blas_mkl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix.dir/blas_mkl.cpp.i"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/may_sparse/xigemm_cpu/matrix/blas_mkl.cpp > CMakeFiles/matrix.dir/blas_mkl.cpp.i

matrix/CMakeFiles/matrix.dir/blas_mkl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix.dir/blas_mkl.cpp.s"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/may_sparse/xigemm_cpu/matrix/blas_mkl.cpp -o CMakeFiles/matrix.dir/blas_mkl.cpp.s

matrix/CMakeFiles/matrix.dir/operator_matrix.cpp.o: matrix/CMakeFiles/matrix.dir/flags.make
matrix/CMakeFiles/matrix.dir/operator_matrix.cpp.o: ../matrix/operator_matrix.cpp
matrix/CMakeFiles/matrix.dir/operator_matrix.cpp.o: matrix/CMakeFiles/matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object matrix/CMakeFiles/matrix.dir/operator_matrix.cpp.o"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT matrix/CMakeFiles/matrix.dir/operator_matrix.cpp.o -MF CMakeFiles/matrix.dir/operator_matrix.cpp.o.d -o CMakeFiles/matrix.dir/operator_matrix.cpp.o -c /root/may_sparse/xigemm_cpu/matrix/operator_matrix.cpp

matrix/CMakeFiles/matrix.dir/operator_matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix.dir/operator_matrix.cpp.i"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/may_sparse/xigemm_cpu/matrix/operator_matrix.cpp > CMakeFiles/matrix.dir/operator_matrix.cpp.i

matrix/CMakeFiles/matrix.dir/operator_matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix.dir/operator_matrix.cpp.s"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/may_sparse/xigemm_cpu/matrix/operator_matrix.cpp -o CMakeFiles/matrix.dir/operator_matrix.cpp.s

matrix/CMakeFiles/matrix.dir/rcxigemm.cpp.o: matrix/CMakeFiles/matrix.dir/flags.make
matrix/CMakeFiles/matrix.dir/rcxigemm.cpp.o: ../matrix/rcxigemm.cpp
matrix/CMakeFiles/matrix.dir/rcxigemm.cpp.o: matrix/CMakeFiles/matrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object matrix/CMakeFiles/matrix.dir/rcxigemm.cpp.o"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT matrix/CMakeFiles/matrix.dir/rcxigemm.cpp.o -MF CMakeFiles/matrix.dir/rcxigemm.cpp.o.d -o CMakeFiles/matrix.dir/rcxigemm.cpp.o -c /root/may_sparse/xigemm_cpu/matrix/rcxigemm.cpp

matrix/CMakeFiles/matrix.dir/rcxigemm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix.dir/rcxigemm.cpp.i"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/may_sparse/xigemm_cpu/matrix/rcxigemm.cpp > CMakeFiles/matrix.dir/rcxigemm.cpp.i

matrix/CMakeFiles/matrix.dir/rcxigemm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix.dir/rcxigemm.cpp.s"
	cd /root/may_sparse/xigemm_cpu/build/matrix && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/may_sparse/xigemm_cpu/matrix/rcxigemm.cpp -o CMakeFiles/matrix.dir/rcxigemm.cpp.s

matrix: matrix/CMakeFiles/matrix.dir/blas_mkl.cpp.o
matrix: matrix/CMakeFiles/matrix.dir/operator_matrix.cpp.o
matrix: matrix/CMakeFiles/matrix.dir/rcxigemm.cpp.o
matrix: matrix/CMakeFiles/matrix.dir/build.make
.PHONY : matrix

# Rule to build all files generated by this target.
matrix/CMakeFiles/matrix.dir/build: matrix
.PHONY : matrix/CMakeFiles/matrix.dir/build

matrix/CMakeFiles/matrix.dir/clean:
	cd /root/may_sparse/xigemm_cpu/build/matrix && $(CMAKE_COMMAND) -P CMakeFiles/matrix.dir/cmake_clean.cmake
.PHONY : matrix/CMakeFiles/matrix.dir/clean

matrix/CMakeFiles/matrix.dir/depend:
	cd /root/may_sparse/xigemm_cpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/may_sparse/xigemm_cpu /root/may_sparse/xigemm_cpu/matrix /root/may_sparse/xigemm_cpu/build /root/may_sparse/xigemm_cpu/build/matrix /root/may_sparse/xigemm_cpu/build/matrix/CMakeFiles/matrix.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : matrix/CMakeFiles/matrix.dir/depend

