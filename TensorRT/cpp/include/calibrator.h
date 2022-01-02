#ifndef CALIBRATOR_HE
#define CALIBRATOR_H

#include "utils.h"


//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!
class Int8EntropyCalibrator: public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const string& dataFileName, const string& tableName, int batch, int width, int height, bool readCache=true);

    void getData();

    int getBatchSize() const noexcept;

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept;

    const void* readCalibrationCache(size_t& length) noexcept;

    void writeCalibrationCache(const void* cache, size_t length) noexcept;

    ~Int8EntropyCalibrator();
private:
    int batch;
    size_t mInputCount;
    void* mDeviceInput{nullptr};

    int totalNum = 0;
    int startIndex = 0;
    int width;
    int height;
    cv::Scalar mean = {0.485, 0.456, 0.406};
    cv::Scalar stdv = {1 /0.229, 1 / 0.224, 1 / 0.225};

    vector<string> dataFiles;
    vector<float> data;

    string mCalibrationFileName;
    string mCalibrationTableName;
    vector<char> mCalibrationCache;
    bool mReadCache;
};

#endif
