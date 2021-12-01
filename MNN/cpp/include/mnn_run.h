#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/opencv.hpp>
#include "MNN/Interpreter.hpp"

using namespace std;
using namespace MNN;


class MNNModel {
public:
    MNNModel(const string& mnnPath, const string& mode="half", const string& device="cpu");

    void forward(const cv::Mat& image, vector<float>& output);

    ~MNNModel();
private:
    string mnnPath;
    string mode;
    string device;
    int numThread = 1;

    Interpreter* net;
    Session* session;
    Tensor* hostInputTensor;
    Tensor* hostOutputTensor;
    Tensor* modelInputTensor;
    Tensor* modelOutputTensor;

    void init();
};
