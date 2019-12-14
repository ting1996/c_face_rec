cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME TestOpenCVCmake)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(/home/pi/Documents/dlib-19.17/dlib dlib_build)

add_subdirectory(/home/pi/Documents/Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn/3rdparty/ncnn ncnn_build)
include_directories(/home/pi/Documents/Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn/3rdparty/ncnn/src)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp UltraFace/UltraFace.cpp)

target_link_libraries(main ${OpenCV_LIBS} dlib::dlib stdc++fs ncnn)

add_executable(face_rec face_rec.cpp)

target_link_libraries(face_rec ${OpenCV_LIBS} dlib::dlib stdc++fs)

add_executable(face_data_webcam face_data_webcam.cpp)

target_link_libraries(face_data_webcam ${OpenCV_LIBS} dlib::dlib stdc++fs)