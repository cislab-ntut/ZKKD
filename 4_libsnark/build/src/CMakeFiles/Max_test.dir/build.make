# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build"

# Include any dependencies generated for this target.
include src/CMakeFiles/Max_test.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/Max_test.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/Max_test.dir/flags.make

src/CMakeFiles/Max_test.dir/test_max.cpp.o: src/CMakeFiles/Max_test.dir/flags.make
src/CMakeFiles/Max_test.dir/test_max.cpp.o: ../src/test_max.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/Max_test.dir/test_max.cpp.o"
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/src" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Max_test.dir/test_max.cpp.o -c "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/test_max.cpp"

src/CMakeFiles/Max_test.dir/test_max.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Max_test.dir/test_max.cpp.i"
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/test_max.cpp" > CMakeFiles/Max_test.dir/test_max.cpp.i

src/CMakeFiles/Max_test.dir/test_max.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Max_test.dir/test_max.cpp.s"
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/test_max.cpp" -o CMakeFiles/Max_test.dir/test_max.cpp.s

src/CMakeFiles/Max_test.dir/test_max.cpp.o.requires:

.PHONY : src/CMakeFiles/Max_test.dir/test_max.cpp.o.requires

src/CMakeFiles/Max_test.dir/test_max.cpp.o.provides: src/CMakeFiles/Max_test.dir/test_max.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/Max_test.dir/build.make src/CMakeFiles/Max_test.dir/test_max.cpp.o.provides.build
.PHONY : src/CMakeFiles/Max_test.dir/test_max.cpp.o.provides

src/CMakeFiles/Max_test.dir/test_max.cpp.o.provides.build: src/CMakeFiles/Max_test.dir/test_max.cpp.o


# Object files for target Max_test
Max_test_OBJECTS = \
"CMakeFiles/Max_test.dir/test_max.cpp.o"

# External object files for target Max_test
Max_test_EXTERNAL_OBJECTS =

src/Max_test: src/CMakeFiles/Max_test.dir/test_max.cpp.o
src/Max_test: src/CMakeFiles/Max_test.dir/build.make
src/Max_test: depends/libsnark/libsnark/libsnark.a
src/Max_test: depends/libsnark/depends/libff/libff/libff.a
src/Max_test: /usr/lib/x86_64-linux-gnu/libgmp.so
src/Max_test: /usr/lib/x86_64-linux-gnu/libgmp.so
src/Max_test: /usr/lib/x86_64-linux-gnu/libgmpxx.so
src/Max_test: src/CMakeFiles/Max_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Max_test"
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/src" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Max_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/Max_test.dir/build: src/Max_test

.PHONY : src/CMakeFiles/Max_test.dir/build

src/CMakeFiles/Max_test.dir/requires: src/CMakeFiles/Max_test.dir/test_max.cpp.o.requires

.PHONY : src/CMakeFiles/Max_test.dir/requires

src/CMakeFiles/Max_test.dir/clean:
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/src" && $(CMAKE_COMMAND) -P CMakeFiles/Max_test.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/Max_test.dir/clean

src/CMakeFiles/Max_test.dir/depend:
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/src" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/build/src/CMakeFiles/Max_test.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : src/CMakeFiles/Max_test.dir/depend

