import os
import onnxruntime
import numpy as np


class ONNXModel:
    def __init__(self, onnx_path, device="cpu"):
        """
        :param onnx_path: local path of onnx file.
        """
        if not os.path.exists(onnx_path):
            raise Exception("model file {} not exist!".format(onnx_path))

        if device == "cpu":
            provider = "CPUExecutionProvider"
        elif device == "gpu":
            provider = "CUDAExecutionProvider"
        else:
            raise Exception("device {} not support!".format(device))

        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=[provider])

        self.input_names = self._get_input_name()
        self.output_names = self._get_output_name()

    def _get_input_name(self):
        """get input node name of onnx model
        """
        input_names = []
        for i, node in enumerate(self.onnx_session.get_inputs()):
            input_names.append(node.name)
            print("input index {}, name: {}, size: {}".format(i, node.name, node.shape))

        return input_names
    
    def _get_output_name(self):
        """get output node name of onnx model
        """
        output_names = []
        for i, node in enumerate(self.onnx_session.get_outputs()):
            output_names.append(node.name)
            print("output index {}, name: {}, size: {}".format(i, node.name, node.shape))

        return output_names

    def forward(self, image_tensors):
        """do infernece
        :param image_tensors: list, inputs of model.
        return outputs: list, outputs of model.
        """
        input_feed = {}
        for i, name in enumerate(self.input_names):
            input_feed[name] = image_tensors[i].astype(np.float32)

        outputs = self.onnx_session.run(self.output_names, input_feed=input_feed)

        return outputs

