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
CMAKE_BINARY_DIR = "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src"

# Include any dependencies generated for this target.
include depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/depend.make

# Include the progress variables for this target.
include depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/progress.make

# Include the compile flags for this target's objects.
include depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/flags.make

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o: depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/flags.make
depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o: ../depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation_example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o"
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/depends/libsnark/depends/libfqfft/tutorials" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o -c "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation_example.cpp"

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.i"
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/depends/libsnark/depends/libfqfft/tutorials" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation_example.cpp" > CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.i

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.s"
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/depends/libsnark/depends/libfqfft/tutorials" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation_example.cpp" -o CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.s

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o.requires:

.PHONY : depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o.requires

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o.provides: depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o.requires
	$(MAKE) -f depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/build.make depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o.provides.build
.PHONY : depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o.provides

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o.provides.build: depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o


# Object files for target lagrange_polynomial_evaluation
lagrange_polynomial_evaluation_OBJECTS = \
"CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o"

# External object files for target lagrange_polynomial_evaluation
lagrange_polynomial_evaluation_EXTERNAL_OBJECTS =

depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation: depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o
depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation: depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/build.make
depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation: depends/libsnark/depends/libff/libff/libff.a
depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation: /usr/lib/x86_64-linux-gnu/libgmp.so
depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation: depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lagrange_polynomial_evaluation"
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/depends/libsnark/depends/libfqfft/tutorials" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lagrange_polynomial_evaluation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/build: depends/libsnark/depends/libfqfft/tutorials/lagrange_polynomial_evaluation

.PHONY : depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/build

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/requires: depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/lagrange_polynomial_evaluation_example.cpp.o.requires

.PHONY : depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/requires

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/clean:
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/depends/libsnark/depends/libfqfft/tutorials" && $(CMAKE_COMMAND) -P CMakeFiles/lagrange_polynomial_evaluation.dir/cmake_clean.cmake
.PHONY : depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/clean

depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/depend:
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/depends/libsnark/depends/libfqfft/tutorials" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/depends/libsnark/depends/libfqfft/tutorials" "/mnt/d/Master_paper/2025_NN to SDT final code/libsnark_testing/test2/libsnark_demo/src/depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : depends/libsnark/depends/libfqfft/tutorials/CMakeFiles/lagrange_polynomial_evaluation.dir/depend

