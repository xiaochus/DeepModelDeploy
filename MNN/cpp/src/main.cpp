#include "mnn_run.h"


int main() {
    // /root/MNN-1.2.0/build/MNNConvert -f ONNX --modelFile net.onnx --MNNModel net.mnn --bizCode biz
    string modelPath = "net.mnn";
    string mode = "fp16";
    string device = "cpu";
    string testImagePath = "/mnt/d/xiaochu/Pictures/005.jpg";

    // model
    MNNModel model(modelPath, mode, device);

    // image preprocess
    cv::Scalar mean = {0.485, 0.456, 0.406};
    cv::Scalar stdv = {1 /0.229, 1 / 0.224, 1 / 0.225};
    cv::Mat img = cv::imread(testImagePath);
    cv::resize(img, img, cv::Size(224, 224));
    img.convertTo(img, CV_32F, 1 / 255.0);
    img = ((img - mean) / stdv);

    vector<float> output;
    model.forward(img, output);
    cout << "output size: " << to_string(output.size()) << endl;

    return 0;
}
