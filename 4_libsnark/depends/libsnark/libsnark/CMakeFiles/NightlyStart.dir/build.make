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

# Utility rule file for NightlyStart.

# Include the progress variables for this target.
include depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/progress.make

depends/libsnark/libsnark/CMakeFiles/NightlyStart:
	cd /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark && /usr/bin/ctest -D NightlyStart

NightlyStart: depends/libsnark/libsnark/CMakeFiles/NightlyStart
NightlyStart: depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/build.make

.PHONY : NightlyStart

# Rule to build all files generated by this target.
depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/build: NightlyStart

.PHONY : depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/build

depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/clean:
	cd /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark && $(CMAKE_COMMAND) -P CMakeFiles/NightlyStart.dir/cmake_clean.cmake
.PHONY : depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/clean

depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/depend:
	cd /home/libsnarkdemo/libsnark_demo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark /home/libsnarkdemo/libsnark_demo/depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : depends/libsnark/libsnark/CMakeFiles/NightlyStart.dir/depend

