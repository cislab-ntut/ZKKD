# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/libsnarkdemo/libsnark_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/libsnarkdemo/libsnark_demo

# Include any dependencies generated for this target.
include depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/depend.make

# Include the progress variables for this target.
include depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/progress.make

# Include the compile flags for this target's objects.
include depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/flags.make

depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.o: depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/flags.make
depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.o: depends/libsnark/libsnark/gadgetlib2/tests/protoboard_UTEST.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/libsnarkdemo/libsnark_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.o"
	cd /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.o -c /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark/gadgetlib2/tests/protoboard_UTEST.cpp

depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.i"
	cd /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark/gadgetlib2/tests/protoboard_UTEST.cpp > CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.i

depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.s"
	cd /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark/gadgetlib2/tests/protoboard_UTEST.cpp -o CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.s

# Object files for target gadgetlib2_protoboard_test
gadgetlib2_protoboard_test_OBJECTS = \
"CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.o"

# External object files for target gadgetlib2_protoboard_test
gadgetlib2_protoboard_test_EXTERNAL_OBJECTS =

depends/libsnark/libsnark/gadgetlib2_protoboard_test: depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/gadgetlib2/tests/protoboard_UTEST.cpp.o
depends/libsnark/libsnark/gadgetlib2_protoboard_test: depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/build.make
depends/libsnark/libsnark/gadgetlib2_protoboard_test: depends/libsnark/libsnark/libsnark.a
depends/libsnark/libsnark/gadgetlib2_protoboard_test: depends/libsnark/depends/gtest/googlemock/gtest/libgtest_main.a
depends/libsnark/libsnark/gadgetlib2_protoboard_test: depends/libsnark/depends/libff/libff/libff.a
depends/libsnark/libsnark/gadgetlib2_protoboard_test: /usr/lib/x86_64-linux-gnu/libgmp.so
depends/libsnark/libsnark/gadgetlib2_protoboard_test: /usr/lib/x86_64-linux-gnu/libgmp.so
depends/libsnark/libsnark/gadgetlib2_protoboard_test: /usr/lib/x86_64-linux-gnu/libgmpxx.so
depends/libsnark/libsnark/gadgetlib2_protoboard_test: depends/libsnark/depends/gtest/googlemock/gtest/libgtest.a
depends/libsnark/libsnark/gadgetlib2_protoboard_test: depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/libsnarkdemo/libsnark_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gadgetlib2_protoboard_test"
	cd /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gadgetlib2_protoboard_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/build: depends/libsnark/libsnark/gadgetlib2_protoboard_test

.PHONY : depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/build

depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/clean:
	cd /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark && $(CMAKE_COMMAND) -P CMakeFiles/gadgetlib2_protoboard_test.dir/cmake_clean.cmake
.PHONY : depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/clean

depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/depend:
	cd /home/libsnarkdemo/libsnark_demo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : depends/libsnark/libsnark/CMakeFiles/gadgetlib2_protoboard_test.dir/depend

