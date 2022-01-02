#include "trt_run.h"


/**
 * @brief Construct a new TRTModel::TRTModel object
 * 
 * @param devId id of cuda device to run.
 * @param modelPath local path to *.onnx model file
 * @param planPath local path to *.plan model file, read/save.
 * @param mode inference mode, fp16/int8
 * @param batchSize max batch_size of model
 * @param useDLACore if use nvidia dla core
 */
TRTModel::TRTModel(int devId, const string& modelPath, const string& planPath, const string &mode, int batchSize, const string& calDataFileName, const string& calTableName, int useDLACoreIndex):
        modelPath(modelPath), planPath(planPath), mode(mode), batchSize(batchSize), useDLACoreIndex(useDLACoreIndex), calDataFileName(calDataFileName), calTableName(calTableName),
        mEngine(nullptr), mStream(nullptr), mContext(nullptr) {
    cudaSetDevice(devId);
    cout << "set cuda device id: " << to_string(devId) << endl;

    bool status = false;

    if (fileExistCheck(this->planPath)) {
        cout << "load trt engine from plan file: " << this->planPath << endl;
        this->loadEngine();
    } else {
        status = this->constructModel();
        assert(status);
    }

    status = this->init();
    assert(status);
}

/**
 * @brief create trt engine from onnx.
 * 
 */
bool TRTModel::constructModel() {
    // create engine builder
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(this->trtLogger));
    if (!builder)
        return false;
    
    // create engine network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
        return false;
    
    // create ONNX parser 
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, this->trtLogger));
    if (!parser)
        return false;
    
    // parse Onnx model and save to network
    auto parsed = parser->parseFromFile(this->modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed)
        return false;
    
    this->checkNetwork(network.get());

    // get engine config from builder
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
        return false;
    
    // config
    builder->setMaxBatchSize(this->batchSize);
    config->setMaxWorkspaceSize(1 << 30);

    // Calibrator life time needs to last until after the engine is built.
    unique_ptr<nvinfer1::IInt8Calibrator> calibrator;

    if (this->mode == "fp16") {
        bool isSupport = builder->platformHasFastFp16();
        if (!isSupport)
            cout << "this device may not fast on fp16 mode." << endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        cout << "set fp16" << endl;
    }

    if (this->mode == "int8") {
        bool isSupport = builder->platformHasFastInt8();
        if (!isSupport) {
            cout << "this device may not fast on int8 mode." << endl;
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            cout << "set fp16" << endl;
        } else {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);

            // set int8 calibrator
            calibrator.reset(new Int8EntropyCalibrator(this->calDataFileName, this->calTableName, this->batchSize, this->inputWidth, this->inputHeight));
            config->setInt8Calibrator(calibrator.get());
            cout << "set int8" << endl;
        }
    }

    // set DLA if use jetson device
    if (this->useDLACoreIndex != -1) {
        cout << "use DLA core index: " << to_string(this->useDLACoreIndex) << endl;
        this->enableDLA(builder.get(), config.get());
    }

    // build trt engine
    SampleUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
        return false;
    
    SampleUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(this->trtLogger)};
    if (!runtime)
        return false;

    this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    if (!this->mEngine)
        return false;

    this->saveEngine();

    return true;
}

/**
 * @brief init trt runtime.
 * 
 */
bool TRTModel::init() {
    // create cuda stream
    CHECK(cudaStreamCreate(&this->mStream));
    // create cuda execute context
    this->mContext = shared_ptr<nvinfer1::IExecutionContext>(this->mEngine->createExecutionContext(), InferDeleter());

    // allocate cuda memory of input node and output node
    for (int i = 0; i < this->mEngine->getNbBindings(); i++) {
        nvinfer1::Dims dims = this->mEngine->getBindingDimensions(i);
        nvinfer1::DataType dtype = this->mEngine->getBindingDataType(i);
        string bindName = this->mEngine->getBindingName(i);

        int64_t elemNum = volume(dims);
        int64_t nbBytes = elemNum * getElementSize(dtype);

        auto *buffer = new CudaBuffer(i, nbBytes, elemNum);
        if (this->mEngine->bindingIsInput(i)) {
            this->inputBind.push_back(buffer);
        } else {
            this->outputBind.push_back(buffer);
        }
    }

    return true;
}

/**
 * @brief check network structure.
 * 
 * @param network trt network.
 */
void TRTModel::checkNetwork(nvinfer1::INetworkDefinition *network) {
    cout << "=== Network Description ===" << endl;

    for (int i = 0; i < network->getNbInputs(); i++) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        this->inputWidth = dims.d[2];
        this->inputHeight = dims.d[1];
        string shape = "(" + to_string(dims.d[0]) + ", " + to_string(dims.d[1]) + ", " + to_string(dims.d[2])  + ", " + to_string(dims.d[3]) + ")" ;
        cout << "Input: " << to_string(i) << " | Name: " << input->getName() << " | shape: " << shape << endl;
    }

    for (int i = 0; i < network->getNbOutputs(); i++) {
        auto output = network->getOutput(i);
        auto dims = output->getDimensions();
        string shape = "(" + to_string(dims.d[0]) + ", " + to_string(dims.d[1]) + ", " + to_string(dims.d[2]) + ")" ;
        cout << "Output: " << to_string(i) << " | Name: " << output->getName() << " | shape: " << shape << endl;
    }
}

/**
 * @brief set dla in trt engine.
 * 
 * @param builder trt builder.
 * @param config trt config.
 * @param allowGPUFallback if allow GPU Fallback
 */
void TRTModel::enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, bool allowGPUFallback) {
    int coreNum = builder->getNbDLACores();

    if (coreNum == 0) {
        cerr << "Trying to use DLA core on a platform that doesn't have any DLA cores" << endl;
        assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
    }

    if (allowGPUFallback) {
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }

    if (!config->getFlag(nvinfer1::BuilderFlag::kINT8)) {
        // User has not requested INT8 Mode.
        // By default run in FP16 mode. FP32 mode is not permitted.
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(this->useDLACoreIndex);
    config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
}

/**
 * @brief load engine from plan file.
 * 
 */
bool TRTModel::loadEngine() {
    bool status = false;

    if (!this->planPath.empty()) {
        ifstream fin(this->planPath);

        // 将文件中的内容读取至字符串
        string cachedEngine;
        while (fin.peek() != EOF) { // 使用fin.peek()防止文件读取时无限循环
            stringstream buffer;
            buffer << fin.rdbuf();
            cachedEngine.append(buffer.str());
        }
        fin.close();

        SampleUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(this->trtLogger)};
        this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(cachedEngine.data(), cachedEngine.size()), InferDeleter());

        if (!this->mEngine)
            status = true;
    }

    return true;
}

/**
 * @brief save engine to plan file.
 * 
 */
bool TRTModel::saveEngine() {
    bool status = false;

    if (!this->planPath.empty()) {
        auto data = SampleUniquePtr<nvinfer1::IHostMemory>(this->mEngine->serialize());

        ofstream serializeOutputStream;

        // 将序列化的模型结果拷贝至serialize_str字符串
        string serializeStr;
        serializeStr.resize(data->size());
        memcpy((void*)serializeStr.data(), data->data(), data->size());

        // 将serialize_str字符串的内容输出至文件
        serializeOutputStream.open(this->planPath);
        serializeOutputStream << serializeStr;
        serializeOutputStream.close();

        status = true;
    }

    return status;
}

/**
 * @brief copy data between host and cuda.
 * 
 */
void TRTModel::copyData(void *dstPtr, const void *srcPtr, int64_t byteSize, cudaMemcpyKind memcpyType, bool async, const cudaStream_t &stream) {
    if (async)
        CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
    else
        CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
}

/**
 * @brief set dla in trt engine.
 * 
 * @param inputData input data.
 * @param outputData output data.
 */
void TRTModel::forward(const vector<float>& inputs, vector<float>& output) {
    // copy data from host to cuda
    vector<void*> inputDataList;
    for (const auto &ib: this->inputBind) {
        this->copyData(ib->getData(), inputs.data(), ib->getBytes(), cudaMemcpyHostToDevice, true, this->mStream);
        inputDataList.emplace_back(ib->getData());
    }
    for (const auto &ob: this->outputBind) {
        inputDataList.emplace_back(ob->getData());
    }

    // inference
    this->mContext->executeV2(inputDataList.data());

    // copy data from cudao to host
    output.resize(this->outputBind[0]->getElements());
    this->copyData(output.data(), this->outputBind[0]->getData(), this->outputBind[0]->getBytes(), cudaMemcpyDeviceToHost, true, this->mStream);
    
    // sync cuda stream
    cudaStreamSynchronize(this->mStream);
}

/**
 * @brief Destroy the TRTModel::TRTModel object
 * 
 */
TRTModel::~TRTModel() {
    cudaStreamSynchronize(this->mStream);
    cudaStreamDestroy(this->mStream);

    for (auto& item: this->inputBind)
        delete item;
    for (auto& item: this->outputBind)
        delete item;
}
