cmake_minimum_required(VERSION 3.17)
project(ONNXModel)

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

set(ONNXRUNTIME_INCLUDE_DIR "/root/onnxruntime-linux-x64-1.9.0/include")
set(ONNXRUNTIME_LIBRARIES "/root/onnxruntime-linux-x64-1.9.0/lib/libonnxruntime.so")

message(STATUS "  OnnxRuntime  libraries: ${ONNXRUNTIME_LIBRARIES}")
message(STATUS "  OnnxRuntime  include path: ${ONNXRUNTIME_INCLUDE_DIR}")

file(GLOB Sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

include_directories(onnx_run ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(onnx_run ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${Sources})

target_include_directories(onnx_run PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
        )

target_link_libraries(onnx_run PUBLIC
        ${ONNXRUNTIME_LIBRARIES}
        ${OpenCV_LIBS}
        )

install(TARGETS onnx_run
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
