#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

#include "trt_run.h"

using namespace std;


int main() {
    string modelPath = "/root/workspace/DeepModelDeploy/net.onnx";
    string planPath = "/root/workspace/DeepModelDeploy/net.plan";
    string testImagePath = "/mnt/d/xiaochu/Pictures/005.jpg";
    // string mode = "fp16";
    string mode = "int8";
    int batch = 1;

    string calDataFileName = "/root/workspace/DeepModelDeploy/dataset.txt";
    string calTableName = "/root/workspace/DeepModelDeploy/calibration.cache";

    // model
    TRTModel model(0, modelPath, planPath, mode, batch, calDataFileName, calTableName);

    // image preprocess
    cv::Scalar mean = {0.485, 0.456, 0.406};
    cv::Scalar stdv = {1 /0.229, 1 / 0.224, 1 / 0.225};
    cv::Mat img = cv::imread(testImagePath);
    cv::resize(img, img, cv::Size(224, 224));
    img.convertTo(img, CV_32F, 1 / 255.0);
    img = img - mean;
    img = img.mul(stdv);

    // hwc 2 chw
    vector<float> inputData;
    vector<cv::Mat> bgrChannels(3);
    cv::split(img, bgrChannels);
    for (auto i = 0; i < bgrChannels.size(); i++) {
        vector<float> data = vector<float>(bgrChannels[i].reshape(1, 1));
        inputData.insert(inputData.end(), data.begin(), data.end());
    }
    vector<float> output;

    auto start = chrono::high_resolution_clock::now();

    model.forward(inputData, output);

    auto end = chrono::high_resolution_clock::now();
    auto total = chrono::duration<double, milli>(end - start).count();
    cout << "total cost: " << to_string(total) << endl;

    cout << "output size: " << to_string(output.size()) << endl;
    cout << "output 0: " << to_string(output[0]) << endl;

    return 0;
}
