cmake_minimum_required(VERSION 3.0)
project(FaceRecognitionDemo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


#find_package( OpenCV REQUIRED )

SET(${OpenCV_INCLUDE_DIRS} C:/Users/Willyzw/Documents/opencv2/build/install/include)
SET(OpenCV_LIBS ${OPENCV_DIR}/lib)

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp)
add_executable(FaceRecognitionDemo ${SOURCE_FILES})

target_link_libraries( FaceRecognitionDemo ${OpenCV_LIBS} )
