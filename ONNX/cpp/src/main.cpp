#include "onnx_run.h"


int main() {
    string modelPath = "/root/workspace/DeepModelDeploy/net.onnx";
    string testImagePath = "/mnt/d/xiaochu/Pictures/005.jpg";

    // model
    ONNXModel model(modelPath);

    // image preprocess
    cv::Scalar mean = {0.485, 0.456, 0.406};
    cv::Scalar stdv = {1 /0.229, 1 / 0.224, 1 / 0.225};
    cv::Mat img = cv::imread(testImagePath);
    cv::resize(img, img, cv::Size(224, 224));
    img.convertTo(img, CV_32F, 1 / 255.0);
    img = img - mean;
    img = img.mul(stdv);
    cv::dnn::blobFromImage(img, img);    // hwc2chw

    vector<float> output;
    model.forward(img, output);
    cout << "output size: " << to_string(output.size()) << endl;

    return 0;
}
