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
include src/CMakeFiles/fc.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/fc.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/fc.dir/flags.make

src/CMakeFiles/fc.dir/fc.cpp.o: src/CMakeFiles/fc.dir/flags.make
src/CMakeFiles/fc.dir/fc.cpp.o: src/fc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/libsnarkdemo/libsnark_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/fc.dir/fc.cpp.o"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fc.dir/fc.cpp.o -c /home/libsnarkdemo/libsnark_demo/src/fc.cpp

src/CMakeFiles/fc.dir/fc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fc.dir/fc.cpp.i"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/libsnarkdemo/libsnark_demo/src/fc.cpp > CMakeFiles/fc.dir/fc.cpp.i

src/CMakeFiles/fc.dir/fc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fc.dir/fc.cpp.s"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/libsnarkdemo/libsnark_demo/src/fc.cpp -o CMakeFiles/fc.dir/fc.cpp.s

# Object files for target fc
fc_OBJECTS = \
"CMakeFiles/fc.dir/fc.cpp.o"

# External object files for target fc
fc_EXTERNAL_OBJECTS =

src/fc: src/CMakeFiles/fc.dir/fc.cpp.o
src/fc: src/CMakeFiles/fc.dir/build.make
src/fc: depends/libsnark/libsnark/libsnark.a
src/fc: depends/libsnark/depends/libff/libff/libff.a
src/fc: /usr/lib/x86_64-linux-gnu/libgmp.so
src/fc: /usr/lib/x86_64-linux-gnu/libgmp.so
src/fc: /usr/lib/x86_64-linux-gnu/libgmpxx.so
src/fc: src/CMakeFiles/fc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/libsnarkdemo/libsnark_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fc"
	cd /home/libsnarkdemo/libsnark_demo/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/fc.dir/build: src/fc

.PHONY : src/CMakeFiles/fc.dir/build

src/CMakeFiles/fc.dir/clean:
	cd /home/libsnarkdemo/libsnark_demo/src && $(CMAKE_COMMAND) -P CMakeFiles/fc.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/fc.dir/clean

src/CMakeFiles/fc.dir/depend:
	cd /home/libsnarkdemo/libsnark_demo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/src /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/src /home/libsnarkdemo/libsnark_demo/src/CMakeFiles/fc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/fc.dir/depend

