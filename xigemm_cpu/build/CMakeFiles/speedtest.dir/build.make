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
include CMakeFiles/speedtest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/speedtest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/speedtest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/speedtest.dir/flags.make

CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o: CMakeFiles/speedtest.dir/flags.make
CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o: ../src/test/speed_test.cpp
CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o: CMakeFiles/speedtest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o -MF CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o.d -o CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o -c /root/may_sparse/xigemm_cpu/src/test/speed_test.cpp

CMakeFiles/speedtest.dir/src/test/speed_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/speedtest.dir/src/test/speed_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/may_sparse/xigemm_cpu/src/test/speed_test.cpp > CMakeFiles/speedtest.dir/src/test/speed_test.cpp.i

CMakeFiles/speedtest.dir/src/test/speed_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/speedtest.dir/src/test/speed_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/may_sparse/xigemm_cpu/src/test/speed_test.cpp -o CMakeFiles/speedtest.dir/src/test/speed_test.cpp.s

# Object files for target speedtest
speedtest_OBJECTS = \
"CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o"

# External object files for target speedtest
speedtest_EXTERNAL_OBJECTS =

speedtest: CMakeFiles/speedtest.dir/src/test/speed_test.cpp.o
speedtest: CMakeFiles/speedtest.dir/build.make
speedtest: CMakeFiles/speedtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/may_sparse/xigemm_cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable speedtest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/speedtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/speedtest.dir/build: speedtest
.PHONY : CMakeFiles/speedtest.dir/build

CMakeFiles/speedtest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/speedtest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/speedtest.dir/clean

CMakeFiles/speedtest.dir/depend:
	cd /root/may_sparse/xigemm_cpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/may_sparse/xigemm_cpu /root/may_sparse/xigemm_cpu /root/may_sparse/xigemm_cpu/build /root/may_sparse/xigemm_cpu/build /root/may_sparse/xigemm_cpu/build/CMakeFiles/speedtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/speedtest.dir/depend

