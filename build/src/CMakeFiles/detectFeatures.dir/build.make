# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/eh420/Documents/RGBD-tutorial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/eh420/Documents/RGBD-tutorial/build

# Include any dependencies generated for this target.
include src/CMakeFiles/detectFeatures.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/detectFeatures.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/detectFeatures.dir/flags.make

src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o: src/CMakeFiles/detectFeatures.dir/flags.make
src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o: ../src/detectFeatures.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eh420/Documents/RGBD-tutorial/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o"
	cd /home/eh420/Documents/RGBD-tutorial/build/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o -c /home/eh420/Documents/RGBD-tutorial/src/detectFeatures.cpp

src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detectFeatures.dir/detectFeatures.cpp.i"
	cd /home/eh420/Documents/RGBD-tutorial/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eh420/Documents/RGBD-tutorial/src/detectFeatures.cpp > CMakeFiles/detectFeatures.dir/detectFeatures.cpp.i

src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detectFeatures.dir/detectFeatures.cpp.s"
	cd /home/eh420/Documents/RGBD-tutorial/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eh420/Documents/RGBD-tutorial/src/detectFeatures.cpp -o CMakeFiles/detectFeatures.dir/detectFeatures.cpp.s

src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o.requires:

.PHONY : src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o.requires

src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o.provides: src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/detectFeatures.dir/build.make src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o.provides.build
.PHONY : src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o.provides

src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o.provides.build: src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o


# Object files for target detectFeatures
detectFeatures_OBJECTS = \
"CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o"

# External object files for target detectFeatures
detectFeatures_EXTERNAL_OBJECTS =

../bin/detectFeatures: src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o
../bin/detectFeatures: src/CMakeFiles/detectFeatures.dir/build.make
../bin/detectFeatures: ../lib/libslambase.a
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stitching3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_superres3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videostab3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_aruco3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bgsegm3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_bioinspired3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ccalib3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_cvv3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dpm3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_face3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_fuzzy3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_hdf3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_img_hash3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_line_descriptor3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_optflow3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_reg3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_rgbd3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_saliency3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_stereo3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_structured_light3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_surface_matching3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_tracking3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xfeatures2d3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ximgproc3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xobjdetect3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_xphoto3.so.3.3.1
../bin/detectFeatures: /usr/local/lib/libpcl_apps.so
../bin/detectFeatures: /usr/local/lib/libpcl_people.so
../bin/detectFeatures: /usr/local/lib/libpcl_outofcore.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libqhull.so
../bin/detectFeatures: /usr/lib/libOpenNI.so
../bin/detectFeatures: /usr/lib/libOpenNI2.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libfreetype.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libz.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libjpeg.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libpng.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libtiff.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingQt-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_photo3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_viz3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_phase_unwrapping3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_datasets3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_plot3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_text3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_dnn3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_ml3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_shape3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_video3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_calib3d3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_features2d3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_flann3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_highgui3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_videoio3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgcodecs3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_objdetect3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_imgproc3.so.3.3.1
../bin/detectFeatures: /opt/ros/kinetic/lib/x86_64-linux-gnu/libopencv_core3.so.3.3.1
../bin/detectFeatures: /usr/local/lib/libpcl_surface.so
../bin/detectFeatures: /usr/local/lib/libpcl_stereo.so
../bin/detectFeatures: /usr/local/lib/libpcl_keypoints.so
../bin/detectFeatures: /usr/local/lib/libpcl_tracking.so
../bin/detectFeatures: /usr/local/lib/libpcl_recognition.so
../bin/detectFeatures: /usr/local/lib/libpcl_registration.so
../bin/detectFeatures: /usr/local/lib/libpcl_segmentation.so
../bin/detectFeatures: /usr/local/lib/libpcl_features.so
../bin/detectFeatures: /usr/local/lib/libpcl_ml.so
../bin/detectFeatures: /usr/local/lib/libpcl_filters.so
../bin/detectFeatures: /usr/local/lib/libpcl_sample_consensus.so
../bin/detectFeatures: /usr/local/lib/libpcl_visualization.so
../bin/detectFeatures: /usr/local/lib/libpcl_io.so
../bin/detectFeatures: /usr/local/lib/libpcl_search.so
../bin/detectFeatures: /usr/local/lib/libpcl_octree.so
../bin/detectFeatures: /usr/local/lib/libpcl_kdtree.so
../bin/detectFeatures: /usr/local/lib/libpcl_common.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libqhull.so
../bin/detectFeatures: /usr/lib/libOpenNI.so
../bin/detectFeatures: /usr/lib/libOpenNI2.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libjpeg.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libpng.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libtiff.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkIOImage-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkIOCore-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkmetaio-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libz.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libXt.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.5.1
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.5.1
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.5.1
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkalglib-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtksys-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libvtkftgl-6.2.so.6.2.0
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libfreetype.so
../bin/detectFeatures: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/detectFeatures: src/CMakeFiles/detectFeatures.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eh420/Documents/RGBD-tutorial/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/detectFeatures"
	cd /home/eh420/Documents/RGBD-tutorial/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/detectFeatures.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/detectFeatures.dir/build: ../bin/detectFeatures

.PHONY : src/CMakeFiles/detectFeatures.dir/build

src/CMakeFiles/detectFeatures.dir/requires: src/CMakeFiles/detectFeatures.dir/detectFeatures.cpp.o.requires

.PHONY : src/CMakeFiles/detectFeatures.dir/requires

src/CMakeFiles/detectFeatures.dir/clean:
	cd /home/eh420/Documents/RGBD-tutorial/build/src && $(CMAKE_COMMAND) -P CMakeFiles/detectFeatures.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/detectFeatures.dir/clean

src/CMakeFiles/detectFeatures.dir/depend:
	cd /home/eh420/Documents/RGBD-tutorial/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eh420/Documents/RGBD-tutorial /home/eh420/Documents/RGBD-tutorial/src /home/eh420/Documents/RGBD-tutorial/build /home/eh420/Documents/RGBD-tutorial/build/src /home/eh420/Documents/RGBD-tutorial/build/src/CMakeFiles/detectFeatures.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/detectFeatures.dir/depend

