CMAKE_MINIMUM_REQUIRED( VERSION 2.8) #Set version
PROJECT( slam) #Set the project name
SET( CMAKE_CXX_COMPILER "g++") #Set the compiler
	
#Set the directory of executable binary files
SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 
	
#Set the directory where the compiled library files are stored
SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 
#And set the directory as the connected directory
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)
	
#Set header file directory
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include)
	
#Add a subfolder, that is, enter the source code folder to continue building
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src)

