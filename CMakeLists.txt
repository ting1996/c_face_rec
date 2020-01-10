cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME TestOpenCVCmake)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

set(MYSQLCONNECTORCPP_ROOT_DIR
        "${MYSQLCONNECTORCPP_ROOT_DIR}"
        CACHE
        PATH
        "Where to start looking for this component.")

find_path(MYSQLCONNECTORCPP_INCLUDE_DIR
        mysql_connection.h
        HINTS
        ${MYSQLCONNECTORCPP_ROOT_DIR}
        PATH_SUFFIXES
        include)

find_library(MYSQLCONNECTORCPP_LIBRARY
        NAMES
        mysqlcppconn
        mysqlcppconn-static
        HINTS
        ${MYSQLCONNECTORCPP_ROOT_DIR}
        PATH_SUFFIXES
        lib64
        lib)

mark_as_advanced(MYSQLCONNECTORCPP_INCLUDE_DIR MYSQLCONNECTORCPP_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MysqlConnectorCpp
        DEFAULT_MSG
        MYSQLCONNECTORCPP_INCLUDE_DIR
        MYSQLCONNECTORCPP_LIBRARY)
        
if(MYSQLCONNECTORCPP_FOUND)
    set(MYSQLCONNECTORCPP_INCLUDE_DIRS "${MYSQLCONNECTORCPP_INCLUDE_DIR}") # Add any dependencies here
    set(MYSQLCONNECTORCPP_LIBRARIES "${MYSQLCONNECTORCPP_LIBRARY}") # Add any dependencies here
    mark_as_advanced(MYSQLCONNECTORCPP_ROOT_DIR)
endif()

add_subdirectory(/home/pi/Documents/dlib/dlib dlib_build)

add_subdirectory(/home/pi/Documents/Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn/3rdparty/ncnn ncnn_build)
include_directories(/home/pi/Documents/Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn/3rdparty/ncnn/src)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp UltraFace/UltraFace.cpp)

target_link_libraries(main ${OpenCV_LIBS} dlib::dlib ncnn ${MYSQLCONNECTORCPP_LIBRARY})

add_executable(face_rec face_rec.cpp)

target_link_libraries(face_rec ${OpenCV_LIBS} dlib::dlib stdc++fs)

add_executable(face_data_webcam face_data_webcam.cpp)

target_link_libraries(face_data_webcam ${OpenCV_LIBS} dlib::dlib stdc++fs)

add_executable(test test.cpp)

target_link_libraries(test ${OpenCV_LIBS} dlib::dlib ${MYSQLCONNECTORCPP_LIBRARY})

set(JetsonGPIO_INCLUDE_DIRS /usr/local/include/)
set(JetsonGPIO_LIBS /usr/local/lib/libJetsonGPIO.a)

include_directories(${JetsonGPIO_INCLUDE_DIRS})
add_executable(test2 test2.cpp)

target_link_libraries(test2  dlib::dlib ${JetsonGPIO_LIBS})


add_executable(run_auto run_auto.cpp UltraFace/UltraFace.cpp)

target_link_libraries(run_auto ${OpenCV_LIBS} dlib::dlib ${MYSQLCONNECTORCPP_LIBRARY} stdc++fs  ncnn ${JetsonGPIO_LIBS}) 
if (DLIB_LINK_WITH_SQLITE3)



endif()
