from utils.utils import process_image

from ONNX.python.onnx_run import ONNXModel
from TensorRT.python.trt_run import TRTModel
from RKNN.python.rknn_run import RKNNMolde
from Mnn.python.mnn_run import MNNModel


if __name__ == '__main__':
    img_path = "0510.jpg"
    img = process_image(img_path, (224, 224))

    # model = ONNXModel("net.onnx", device="gpu")
    # output = model.forward([img])
    # print(output[0].shape)

    # model = TRTModel("net.onnx", "net.plan", "fp16")
    # output = model.forward([img])
    # print(output[0].shape)

    # model = RKNNMolde("net.onnx", "net.rknn", "hybrid", quant_data_file="dataset.txt")
    # output = model.forward([img])
    # print(output[0].shape)
    # model.perf([img])

    model = MNNModel("net.mnn")
    output = model.forward([img])
    print(output[0].shape)
