#include "mnn_run.h"


/**
 * @brief Construct a new MNNModel::MNNModel object
 * 
 * @param mnnPath local path to *.mnn model file
 * @param mode inference mode, fp32/fp16/half
 * @param device inference device and driver, cpu/opencl/opengl/vulkan/metal
 */
MNNModel::MNNModel(const string& mnnPath, const string& mode, const string& device):
    mnnPath(mnnPath), mode(mode), device(device) {

    this->init();
}

/**
 * @brief init mnn network
 * 
 */
void MNNModel::init() {
    struct stat buffer;
    if (stat(this->mnnPath.c_str(), &buffer) != 0) {
        cout << "model file: " << this->mnnPath << " not exists!" << endl; 
        assert(0);
    }

    // build network
    this->net = Interpreter::createFromFile(this->mnnPath.c_str());

    // build config
    ScheduleConfig config;

    // set cpu thread used
    config.numThread = this->numThread;

    // set host device
    if (this->device == "cpu") {
        config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    } else if (this->device == "opencl") {
        config.type = static_cast<MNNForwardType>(MNN_FORWARD_OPENCL);
    } else if (this->device == "opengl") {
        config.type = static_cast<MNNForwardType>(MNN_FORWARD_OPENGL);
    } else if (this->device == "vulkan") {
        config.type = static_cast<MNNForwardType>(MNN_FORWARD_VULKAN);
    } else if (this->device == "metal") {
        config.type = static_cast<MNNForwardType>(MNN_FORWARD_METAL);
    } else {
        cout << "Un-support device: " << this->device << endl; 
        assert(0);
    }

    // set precision
    BackendConfig backendConfig;
    if (this->mode == "fp16") {
        backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(BackendConfig::Precision_Low);
    } else if (this->mode == "half") {
        backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(BackendConfig::Precision_Normal);
    } else if (this->mode == "fp32") {
        backendConfig.precision = static_cast<BackendConfig::PrecisionMode>(BackendConfig::Precision_High);
    } else {
        cout << "Un-support mode: " << this->mode << endl; 
        assert(0);
    }

    // set power use
    backendConfig.power = static_cast<BackendConfig::PowerMode>(BackendConfig::Power_Normal);
    // set memory use
    backendConfig.memory = static_cast<BackendConfig::MemoryMode>(BackendConfig::Memory_Normal);

    config.backendConfig = &backendConfig;

    // build session use config
    this->session = this->net->createSession(config);

    // get input and output node of network
    this->modelInputTensor = this->net->getSessionInput(this->session, NULL);
    this->modelOutputTensor = this->net->getSessionOutput(this->session, NULL);
    this->hostInputTensor = new Tensor(this->modelInputTensor, Tensor::CAFFE);
    this->hostOutputTensor = new Tensor(this->modelOutputTensor, Tensor::CAFFE);
}

/**
 * @brief inference on mnn
 * 
 * @param image opencv image
 * @param output output float data
 */
void MNNModel::forawrd(const cv::Mat& image, vector<float>& output) {
    memcpy(this->hostInputTensor->host<float>(), image.data, this->hostInputTensor->size());

    this->modelInputTensor->copyFromHostTensor(this->hostInputTensor);
    net->runSession(session);
    this->modelOutputTensor->copyToHostTensor(this->hostOutputTensor);

    output.resize(this->hostOutputTensor->elementSize());
    memcpy(output.data(), this->hostOutputTensor->host<float>(), this->hostOutputTensor->size());
}

/**
 * @brief Destroy the MNNModel::MNNModel object
 * 
 */
MNNModel::~MNNModel() {
    delete this->hostInputTensor;
    delete this->hostOutputTensor;

    if (this->net != nullptr)
        delete this->net;
}
