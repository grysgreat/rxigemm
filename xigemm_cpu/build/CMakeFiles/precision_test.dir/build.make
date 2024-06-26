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
include CMakeFiles/precision_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/precision_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/precision_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/precision_test.dir/flags.make

CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o: CMakeFiles/precision_test.dir/flags.make
CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o: ../src/test/rcxigemm_test.cpp
CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o: CMakeFiles/precision_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o -MF CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o.d -o CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o -c /root/may_sparse/xigemm_cpu/src/test/rcxigemm_test.cpp

CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/may_sparse/xigemm_cpu/src/test/rcxigemm_test.cpp > CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.i

CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/may_sparse/xigemm_cpu/src/test/rcxigemm_test.cpp -o CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.s

# Object files for target precision_test
precision_test_OBJECTS = \
"CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o"

# External object files for target precision_test
precision_test_EXTERNAL_OBJECTS =

precision_test: CMakeFiles/precision_test.dir/src/test/rcxigemm_test.cpp.o
precision_test: CMakeFiles/precision_test.dir/build.make
precision_test: /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_core.a
precision_test: /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_gnu_thread.a
precision_test: /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_intel_ilp64.a
precision_test: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
precision_test: /usr/lib/x86_64-linux-gnu/libpthread.so
precision_test: CMakeFiles/precision_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable precision_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/precision_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/precision_test.dir/build: precision_test
.PHONY : CMakeFiles/precision_test.dir/build

CMakeFiles/precision_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/precision_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/precision_test.dir/clean

CMakeFiles/precision_test.dir/depend:
	cd /root/may_sparse/xigemm_cpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/may_sparse/xigemm_cpu /root/may_sparse/xigemm_cpu /root/may_sparse/xigemm_cpu/build /root/may_sparse/xigemm_cpu/build /root/may_sparse/xigemm_cpu/build/CMakeFiles/precision_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/precision_test.dir/depend

