#ifndef UTILS_H
#define UTILS_H

#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"

using namespace std;


#define CHECK(status)                                                                                                  \
    do {                                                                                                               \
        auto ret = (status);                                                                                           \
        if (ret != 0) {                                                                                                \
            cerr << "Cuda failure: " << ret << endl;                                                                   \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

/**
 * overwrite log class
 */
class Logger : public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            cout << msg << endl;
    }
};

/**
 * safe pointer of engine
 */
struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using SampleUniquePtr = unique_ptr<T, InferDeleter>;


/**
 * cuda bind buffer
 */
class CudaBuffer {
public:
    CudaBuffer(): bufferIndex(0), nbBytes(0), nbElem(0), bufferData(nullptr) {}

    CudaBuffer(int index, int64_t bytes, int64_t nbElem): bufferIndex(index), nbBytes(bytes), nbElem(nbElem), bufferData(nullptr) {
        CHECK(cudaMalloc(&this->bufferData, this->nbBytes));
        if (this->bufferData == nullptr) {
            cout << "cuda assign out of memory" << endl;
        }
    }

    void* getData() {
        return this->bufferData;
    }

    int64_t getBytes() const {
        return this->nbBytes;
    }

    int64_t getElements() const {
        return this->nbElem;
    }

    ~CudaBuffer() {
        cudaFree(this->bufferData);
    }
private:
    int bufferIndex;
    int64_t nbBytes;
    int64_t nbElem;
    void* bufferData;
};


/**
 * get byte size of datatype
 */
inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
    switch (t) {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }

    return 0;
}

/**
 * get size of dim
 */
inline int64_t volume(const nvinfer1::Dims& d) {
    return accumulate(d.d, d.d + d.nbDims, 1, multiplies<int64_t>());
}


/**
 * @brief check file exist status.
 */
static bool fileExistCheck(const string& filePath) {
    bool status = true;

    struct stat buffer;
    if (stat(filePath.c_str(), &buffer) != 0) {
        status = false;
    }

    return status;
};

# endif