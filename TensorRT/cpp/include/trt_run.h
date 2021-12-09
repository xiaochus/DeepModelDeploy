#include <memory>
#include <cassert>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>

#include "utils.h"


class TRTModel {
public:
    TRTModel(int devId, const string& modelPath, const string& planPath, const string &mode, int batchSize, int useDLACoreIndex=-1);

    void forward(const vector<float>& inputs, vector<float>& output);

    ~TRTModel();
private:
    Logger trtLogger;

    int batchSize;
    int useDLACoreIndex;
    string modelPath;
    string planPath;
    string mode;

    cudaStream_t mStream;
    shared_ptr<nvinfer1::ICudaEngine> mEngine;
    shared_ptr<nvinfer1::IExecutionContext> mContext;

    vector<CudaBuffer*> inputBind;
    vector<CudaBuffer*> outputBind;

    bool init();

    bool constructModel();

    void checkNetwork(nvinfer1::INetworkDefinition *network);

    void enableDLA(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, bool allowGPUFallback = true);

    bool loadEngine();

    bool saveEngine();

    bool fileExistCheck(const string& filePath);

    void copyData(void *dstPtr, const void *srcPtr, int64_t byteSize, cudaMemcpyKind memcpyType, bool async=true, const cudaStream_t &stream=nullptr);
};
