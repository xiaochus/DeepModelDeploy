cmake_minimum_required(VERSION 3.17)
project(TRTModel)

# ignore trt warning
add_definitions(-w)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -O3
        -gencode arch=compute_86,code=sm_86
)

set(TENSORRT_ROOT "/root/TensorRT-8.0.3.4/")
set(CUDA_PATH "/usr/local/cuda")

# Opencv
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

# CUDA
find_package(CUDA REQUIRED)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

if(NOT CUDA_LIBRARY_DIRS)
    set(CUDA_LIBRARY_DIRS ${CUDA_PATH}/lib/x64)
    message(WARING " Can not find CUDA lib. It will use the default path => ${CUDA_LIBRARY_DIRS}")
endif()

message(STATUS "CUDA_INCLUDE_DIRS : ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARY_DIRS : ${CUDA_LIBRARY_DIRS}")


# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_ONNX nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_INFER_ONNX})
message(STATUS "Found TensorRT libs at ${TENSORRT_LIBRARY}")

find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)

if(NOT TENSORRT_FOUND)
    message(ERROR "Cannot find TensorRT library.")
endif()

# build
file(GLOB Sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

include_directories(trt_run ${CMAKE_CURRENT_SOURCE_DIR}/include)

cuda_add_executable(trt_run ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${Sources})

target_include_directories(trt_run PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_run
        ${TENSORRT_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cudart_static_LIBRARY}
        ${OpenCV_LIBS})

install(TARGETS trt_run
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
