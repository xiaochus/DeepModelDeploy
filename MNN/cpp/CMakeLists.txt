cmake_minimum_required(VERSION 3.17)
project(MNNModel)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Error! OpenCV not found!")

    set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
    set(OpenCV_LIBS "/usr/local/lib")
endif()

set(MNN_INCLUDE_DIR "/root/MNN-1.2.0/include")
set(MNN_LIBRARIES "/root/MNN-1.2.0/build/libMNN.so")

message(STATUS "  MNN  libraries: ${MNN_LIBRARIES}")
message(STATUS "  MNN  include path: ${MNN_INCLUDE_DIR}")

file(GLOB Sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

include_directories(mnn_run ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(mnn_run ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${Sources})

target_include_directories(mnn_run PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${MNN_INCLUDE_DIR}
        )

target_link_libraries(mnn_run PUBLIC
        ${MNN_LIBRARIES}
        ${OpenCV_LIBS}
        )

install(TARGETS mnn_run
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
