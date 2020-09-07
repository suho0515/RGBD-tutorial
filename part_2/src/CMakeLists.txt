# Add an executable binary
ADD_EXECUTABLE( main main.cpp )

# Dependent increase in PCL library
FIND_PACKAGE (PCL REQUIRED )

list ( REMOVE_ITEM PCL_LIBRARIES "vtkproj4" ) # use this in Ubuntu 16.04
# Increased reliance on the opencv
FIND_PACKAGE (OpenCV REQUIRED )

#Add header files and library files
ADD_DEFINITIONS ( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES ( ${PCL_INCLUDE_DIRS}   )
LINK_LIBRARIES ( ${PCL_LIBRARY_DIRS} )

ADD_EXECUTABLE (generate_pointcloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES (generate_pointcloud ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


