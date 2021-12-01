# DeepModelDeploy

Deploy deep learning model on difference hardware and framework. 

This repo provide some deploy cases by python and cpp. Generally, we test/eval/simulate the model by python on PC and deploy the model by cpp on device.

*Current support hardware:* `x86 CPU\Nvidia GPU\ARM CPU\ARM GPU\ARM RKNN NPU`.

*Current support framework:* `OnnxRuntime\TensorRT\MNN\RKNN-Toolkit`.

**still under developing...**

## Requirement

### python
- Python 3.7
- opencv 4.5
- pytorch 1.10 
- onnxruntime/onnxruntime-gpu 1.9.0
- rknn-toolkit 1.7.1
- MNN 1.1.6
### cpp
- CMake 3.19.1
- OpenCV 4.5.0
- TensorRT 8.0.3.4
- MNN 1.2.0

## Acknowledgement

- [Pytorch](https://github.com/pytorch/pytorch)
- [TensorRT](https://github.com/NVIDIA/TensorRT)
- [OnnxRuntime](https://github.com/microsoft/onnxruntime)
- [MNN](https://github.com/alibaba/MNN)
- [rknpu](https://github.com/rockchip-linux/rknpu)
- [rknn-toolkit](https://github.com/rockchip-linux/rknn-toolkit)