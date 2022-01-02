#include <iterator>

#include "calibrator.h"


Int8EntropyCalibrator::Int8EntropyCalibrator(const string& dataFileName, const string& tableName, int batch, int width, int height, bool readCache)
        : mCalibrationFileName(dataFileName), batch(batch), width(width), height(height), mCalibrationTableName(tableName), mReadCache(readCache) {
    this->getData();
    this->mInputCount = this->width * this->height * 3 * this->batch;
    CHECK(cudaMalloc(&this->mDeviceInput, this->mInputCount * sizeof(float)));
}

void Int8EntropyCalibrator::getData() {
    ifstream fin(this->mCalibrationFileName);
    string temp;
    while (fin >> temp) {
        this->dataFiles.push_back(temp);
    }

    this->totalNum = this->dataFiles.size();
}

int Int8EntropyCalibrator::getBatchSize() const noexcept {
    return this->batch;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if ((this->startIndex + this->batch) >= this->totalNum) {
        return false;
    }

    this->data.clear();
    cv::Mat img;

    for (int i = 0; i < this->batch; i++) {
        img = cv::imread(this->dataFiles[this->startIndex + i]);
        cv::resize(img, img, cv::Size(this->width, this->height));
        img.convertTo(img, CV_32F, 1 / 255.0);
        img = img - this->mean;
        img = img.mul(this->stdv);

        // hwc 2 chw
        vector<cv::Mat> bgrChannels(3);
        cv::split(img, bgrChannels);
        for (auto i = 0; i < bgrChannels.size(); i++) {
            vector<float> tmpData = vector<float>(bgrChannels[i].reshape(1, 1));
            this->data.insert(this->data.end(), tmpData.begin(), tmpData.end());
        }
    }

    CHECK(cudaMemcpy(this->mDeviceInput, this->data.data(), this->mInputCount * sizeof(float), cudaMemcpyHostToDevice));
    bindings[0] = this->mDeviceInput;

    this->startIndex += this->batch;

    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    cout << "read calibrationCache: " << this->mCalibrationTableName << endl;
    this->mCalibrationCache.clear();
 
    ifstream input(this->mCalibrationTableName.c_str(), ios::binary);
    input >> noskipws;

    if (this->mReadCache && input.good()) {
        copy(istream_iterator<char>(input), istream_iterator<char>(), back_inserter(mCalibrationCache));
    }

    length = this->mCalibrationCache.size();
    return length ? this->mCalibrationCache.data() : nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    cout << "write calibrationCache: " << this->mCalibrationTableName << endl;

    ofstream output(this->mCalibrationTableName.c_str(), ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    CHECK(cudaFree(this->mDeviceInput));
}
