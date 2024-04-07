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
include CMakeFiles/rcxigemm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/rcxigemm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/rcxigemm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rcxigemm.dir/flags.make

CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o: CMakeFiles/rcxigemm.dir/flags.make
CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o: ../src/test/precision_test.cpp
CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o: CMakeFiles/rcxigemm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o -MF CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o.d -o CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o -c /root/may_sparse/xigemm_cpu/src/test/precision_test.cpp

CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/may_sparse/xigemm_cpu/src/test/precision_test.cpp > CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.i

CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/may_sparse/xigemm_cpu/src/test/precision_test.cpp -o CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.s

# Object files for target rcxigemm
rcxigemm_OBJECTS = \
"CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o"

# External object files for target rcxigemm
rcxigemm_EXTERNAL_OBJECTS =

rcxigemm: CMakeFiles/rcxigemm.dir/src/test/precision_test.cpp.o
rcxigemm: CMakeFiles/rcxigemm.dir/build.make
rcxigemm: /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_core.a
rcxigemm: /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_gnu_thread.a
rcxigemm: /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_intel_ilp64.a
rcxigemm: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
rcxigemm: /usr/lib/x86_64-linux-gnu/libpthread.so
rcxigemm: CMakeFiles/rcxigemm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rcxigemm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rcxigemm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rcxigemm.dir/build: rcxigemm
.PHONY : CMakeFiles/rcxigemm.dir/build

CMakeFiles/rcxigemm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rcxigemm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rcxigemm.dir/clean

CMakeFiles/rcxigemm.dir/depend:
	cd /root/may_sparse/xigemm_cpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/may_sparse/xigemm_cpu /root/may_sparse/xigemm_cpu /root/may_sparse/xigemm_cpu/build /root/may_sparse/xigemm_cpu/build /root/may_sparse/xigemm_cpu/build/CMakeFiles/rcxigemm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rcxigemm.dir/depend

