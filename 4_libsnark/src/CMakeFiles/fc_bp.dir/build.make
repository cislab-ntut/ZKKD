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
include src/CMakeFiles/fc_bp.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/fc_bp.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/fc_bp.dir/flags.make

src/CMakeFiles/fc_bp.dir/fc_bp.cpp.o: src/CMakeFiles/fc_bp.dir/flags.make
src/CMakeFiles/fc_bp.dir/fc_bp.cpp.o: src/fc_bp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/libsnarkdemo/libsnark_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/fc_bp.dir/fc_bp.cpp.o"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fc_bp.dir/fc_bp.cpp.o -c /home/libsnarkdemo/libsnark_demo/src/fc_bp.cpp

src/CMakeFiles/fc_bp.dir/fc_bp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fc_bp.dir/fc_bp.cpp.i"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/libsnarkdemo/libsnark_demo/src/fc_bp.cpp > CMakeFiles/fc_bp.dir/fc_bp.cpp.i

src/CMakeFiles/fc_bp.dir/fc_bp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fc_bp.dir/fc_bp.cpp.s"
	cd /home/libsnarkdemo/libsnark_demo/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/libsnarkdemo/libsnark_demo/src/fc_bp.cpp -o CMakeFiles/fc_bp.dir/fc_bp.cpp.s

# Object files for target fc_bp
fc_bp_OBJECTS = \
"CMakeFiles/fc_bp.dir/fc_bp.cpp.o"

# External object files for target fc_bp
fc_bp_EXTERNAL_OBJECTS =

src/fc_bp: src/CMakeFiles/fc_bp.dir/fc_bp.cpp.o
src/fc_bp: src/CMakeFiles/fc_bp.dir/build.make
src/fc_bp: depends/libsnark/libsnark/libsnark.a
src/fc_bp: depends/libsnark/depends/libff/libff/libff.a
src/fc_bp: /usr/lib/x86_64-linux-gnu/libgmp.so
src/fc_bp: /usr/lib/x86_64-linux-gnu/libgmp.so
src/fc_bp: /usr/lib/x86_64-linux-gnu/libgmpxx.so
src/fc_bp: src/CMakeFiles/fc_bp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/libsnarkdemo/libsnark_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fc_bp"
	cd /home/libsnarkdemo/libsnark_demo/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fc_bp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/fc_bp.dir/build: src/fc_bp

.PHONY : src/CMakeFiles/fc_bp.dir/build

src/CMakeFiles/fc_bp.dir/clean:
	cd /home/libsnarkdemo/libsnark_demo/src && $(CMAKE_COMMAND) -P CMakeFiles/fc_bp.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/fc_bp.dir/clean

src/CMakeFiles/fc_bp.dir/depend:
	cd /home/libsnarkdemo/libsnark_demo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/src /home/libsnarkdemo/libsnark_demo /home/libsnarkdemo/libsnark_demo/src /home/libsnarkdemo/libsnark_demo/src/CMakeFiles/fc_bp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/fc_bp.dir/depend

