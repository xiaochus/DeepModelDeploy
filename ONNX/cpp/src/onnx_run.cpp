#include "onnx_run.h"


/**
 * @brief Construct a new ONNXModel::ONNXModel object
 * 
 * @param onnxPath local path to *.onnx model file
 */
ONNXModel::ONNXModel(const string& onnxPath): onnxPath(onnxPath), session(nullptr) {
    this->init();
}

/**
 * @brief init onnxruntime network
 * 
 */
void ONNXModel::init() {
    struct stat buffer;
    if (stat(this->onnxPath.c_str(), &buffer) != 0) {
        cout << "model file: " << this->onnxPath << " not exists!" << endl; 
        assert(0);
    }

    string instanceName = "onnx_run_demo";

    // init session
    this->env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    this->sessionOptions.SetIntraOpNumThreads(1);
    this->session = Ort::Session(this->env, onnxPath.c_str(), sessionOptions);

    // get input/output nodes and shapes
    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    cout << "Number of Input Nodes: " << numInputNodes << endl;
    for (size_t i = 0; i < numInputNodes; i++) {
        const char* inputName = session.GetInputName(i, allocator);

        Ort::TypeInfo inputTypeInfo = this->session.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
        vector<int64_t> inputDims = inputTensorInfo.GetShape();

        this->inputNames.push_back(inputName);
        this->inputShapes.push_back(inputDims);
        int elem = 1;

        cout << "Input Name: " << inputName << ", type: " << inputType << ", shapes: (";
        for (int j = 0; j < inputDims.size(); j++) {
            elem *= inputDims[j];
            cout << inputDims[j] << ", ";
        }
        cout << ")" << endl;

        this->inputElements.push_back(elem);
    }

    cout << "Number of Output Nodes: " << numOutputNodes << endl;
    for (size_t i = 0; i < numOutputNodes; i++) {
        const char* outputName = session.GetOutputName(i, allocator);

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
        vector<int64_t> outputDims = outputTensorInfo.GetShape();

        this->outputNames.push_back(outputName);
        this->outputShapes.push_back(outputDims);
        int elem = 1;

        cout << "Output Name: " << outputName << ", type: " << outputType << ", shapes: (";
        for (int j = 0; j < outputDims.size(); j++) {
            elem *= outputDims[j];
            cout << outputDims[j] << ", ";
        }
        cout << ")" << endl;

        this->outputElements.push_back(elem);
    }
}

/**
 * @brief inference on onnx
 * 
 * @param image opencv image
 * @param output output float data
 */
void ONNXModel::forward(const cv::Mat& image, vector<float>& output) {
    this->inputTensors.clear();
    this->outputTensors.clear();
    output.resize(this->outputElements[0]);

    // copy data to onnx tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    this->inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)image.data, this->inputElements[0], 
                                                                 this->inputShapes[0].data(), this->inputShapes[0].size()));

    this->outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, output.data(), this->outputElements[0], 
                                                                  this->outputShapes[0].data(), this->outputShapes[0].size()));
    // run session
    this->session.Run(Ort::RunOptions{nullptr}, 
                      this->inputNames.data(), this->inputTensors.data(), 1, 
                      this->outputNames.data(), this->outputTensors.data(), 1);
}

/**
 * @brief Destroy the ONNXModel::ONNXModel object
 * 
 */
ONNXModel::~ONNXModel() {
}
