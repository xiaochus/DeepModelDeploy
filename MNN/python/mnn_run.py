import os
import MNN
import numpy as np


class MNNModel:
    def __init__(self, model_path, device="CPU"):
        """
        :param model_path: local path of mnn file.
        :param device: run device, CPU only for pypi.
        """
        if not os.path.exists(model_path):
            raise Exception("model file {} not exist!".format(model_path))

        self.device = device
        self._get_session(model_path)

    def _get_session(self, model_path):
        """build run session
        :param model_path: local path of mnn file.
        """
        self.interpreter = MNN.Interpreter(model_path)

        # config
        config = {}
        config['backend'] = self.device
        config['precision'] = 'low'
        config['numThread'] = 1

        # create session
        self.session = self.interpreter.createSession(config)
        self.interpreter.resizeSession(self.session)

        # input and output nodes of model
        self.input_tensors = self.interpreter.getSessionInputAll(self.session)
        self.output_tensors = self.interpreter.getSessionOutputAll(self.session)

        for i, (k, v) in enumerate(self.input_tensors.items()):
            print("input index: {}, name: {}, shape: {}".format(i, k, v.getShape()))
        for i, (k, v) in enumerate(self.output_tensors.items()):
            print("output index: {}, name: {}, shape: {}".format(i, k, v.getShape()))

    def forward(self, image_tensors):
        """do infernece
        :param image_tensors: list, inputs of model.
        return outputs: list, outputs of model.
        """
        outputs = []

        # copy from host to device
        for i, (k, v) in enumerate(self.input_tensors.items()):
            image = image_tensors[i].astype(np.float32)
            tmp_input = MNN.Tensor(v.getShape(), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
            v.copyFrom(tmp_input)

        # run
        self.interpreter.runSession(self.session)

        # copy from device to host
        for i, (k, v) in enumerate(self.output_tensors.items()):
            tmp_output = MNN.Tensor(v.getShape(), MNN.Halide_Type_Float, np.ones(v.getShape()).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
            v.copyToHostTensor(tmp_output)
            outputs.append(np.array(tmp_output.getData()))

        return outputs
