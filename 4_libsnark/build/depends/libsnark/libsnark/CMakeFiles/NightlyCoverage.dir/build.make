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
CMAKE_SOURCE_DIR = "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark/build"

# Utility rule file for NightlyCoverage.

# Include the progress variables for this target.
include depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/progress.make

depends/libsnark/libsnark/CMakeFiles/NightlyCoverage:
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark/build/depends/libsnark/libsnark" && /usr/bin/ctest -D NightlyCoverage

NightlyCoverage: depends/libsnark/libsnark/CMakeFiles/NightlyCoverage
NightlyCoverage: depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/build.make

.PHONY : NightlyCoverage

# Rule to build all files generated by this target.
depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/build: NightlyCoverage

.PHONY : depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/build

depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/clean:
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark/build/depends/libsnark/libsnark" && $(CMAKE_COMMAND) -P CMakeFiles/NightlyCoverage.dir/cmake_clean.cmake
.PHONY : depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/clean

depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/depend:
	cd "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark" "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark/depends/libsnark/libsnark" "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark/build" "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark/build/depends/libsnark/libsnark" "/mnt/d/Master_paper/2025_NN to SDT final code/upload/4_libsnark/build/depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : depends/libsnark/libsnark/CMakeFiles/NightlyCoverage.dir/depend

