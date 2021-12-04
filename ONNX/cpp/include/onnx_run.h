#include <string>
#include <vector>
#include <cassert>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "onnxruntime_cxx_api.h"


using namespace std;


class ONNXModel {
public:
    ONNXModel(const string& onnxPath);

    void forward(const cv::Mat& image, vector<float>& output);

    ~ONNXModel();
private:
    string onnxPath;

    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    vector<const char*> inputNames;
    vector<const char*> outputNames;
    vector<vector<int64_t>> inputShapes;
    vector<vector<int64_t>> outputShapes;
    vector<int> inputElements;
    vector<int> outputElements;
    vector<Ort::Value> inputTensors;
    vector<Ort::Value> outputTensors;

    void init();
};
