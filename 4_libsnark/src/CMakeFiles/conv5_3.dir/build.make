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
include src/CMakeFiles/conv5_3.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/conv5_3.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/conv5_3.dir/flags.make

src/CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.o: src/CMakeFiles/conv5_3.dir/flags.make
src/CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.o: src/vgg16/circuit/conv5_3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/libsnarkdemo/libsnark_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.o"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.o -c /home/libsnarkdemo/libsnark_demo/src/vgg16/circuit/conv5_3.cpp

src/CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.i"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/libsnarkdemo/libsnark_demo/src/vgg16/circuit/conv5_3.cpp > CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.i

src/CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.s"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/libsnarkdemo/libsnark_demo/src/vgg16/circuit/conv5_3.cpp -o CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.s

# Object files for target conv5_3
conv5_3_OBJECTS = \
"CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.o"

# External object files for target conv5_3
conv5_3_EXTERNAL_OBJECTS =

vgg16/conv5_3: src/CMakeFiles/conv5_3.dir/vgg16/circuit/conv5_3.cpp.o
vgg16/conv5_3: src/CMakeFiles/conv5_3.dir/build.make
vgg16/conv5_3: depends/libsnark/libsnark/libsnark.a
vgg16/conv5_3: depends/libsnark/depends/libff/libff/libff.a
vgg16/conv5_3: /usr/lib/x86_64-linux-gnu/libgmp.so
vgg16/conv5_3: /usr/lib/x86_64-linux-gnu/libgmp.so
vgg16/conv5_3: /usr/lib/x86_64-linux-gnu/libgmpxx.so
vgg16/conv5_3: src/CMakeFiles/conv5_3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/libsnarkdemo/libsnark_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../vgg16/conv5_3"
	cd /home/libsnarkdemo/libsnark_demo/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/conv5_3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/conv5_3.dir/build: vgg16/conv5_3

.PHONY : src/CMakeFiles/conv5_3.dir/build

src/CMakeFiles/conv5_3.dir/clean:
	cd /home/libsnarkdemo/libsnark_demo/src && $(CMAKE_COMMAND) -P CMakeFiles/conv5_3.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/conv5_3.dir/clean

src/CMakeFiles/conv5_3.dir/depend:
	cd /home/libsnarkdemo/libsnark_demo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/src /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/src /home/libsnarkdemo/libsnark_demo/src/CMakeFiles/conv5_3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/conv5_3.dir/depend

