import os
import numpy as np
import tensorrt as trt

from .utils import common, calibrator


class TRTModel:
    def __init__(self, onnx_path, plan_path, mode="fp16"):
        """
        :param onnx_path: local path of onnx file.
        :param plan_path: trt plan file to read/save.
        :param mode: inference mode, fp16/int8.
        """
        self.trt_logger = trt.Logger()

        self.onnx_path = onnx_path
        self.plan_path = plan_path
        self.mode = mode

        self.parse_input_shape = []
        self.parse_output_shape= []

        self.engine = self._get_engine()
        self.execution_context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def _check_network(self, network):
        """check network
        :param network: INetworkDefinition
        """
        if not network.num_outputs:
            raise Exception("No output node found!")

        input_nodes = [network.get_input(i) for i in range(network.num_inputs)]
        output_nodes = [network.get_output(i) for i in range(network.num_outputs)]

        print("Network description")
        for i, inp in enumerate(input_nodes):
            print("Input node {} | Name {} | Shape {}".format(i, inp.name, inp.shape))

        print("Total layers: {}".format(network.num_layers))
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            print("index {}, layer name: {}".format(i, layer.name))

        for i, out in enumerate(output_nodes):
            print("Output node {} | Name {} | Shape {}".format(i, out.name, out.shape))

    def _parse_onnx(self):
        """takes an ONNX file and creates a TensorRT engine to run inference with
        """
        dynamic = False
        flag = common.EXPLICIT_BATCH

        with trt.Builder(self.trt_logger) as builder, builder.create_network(flag) as network, builder.create_builder_config() as config, trt.OnnxParser(network, self.trt_logger) as parser, trt.Runtime(self.trt_logger) as runtime:
            config.max_workspace_size = common.GiB(1)
            builder.max_batch_size = 1

            if self.mode == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                print("set FP16 mode.")
            if self.mode == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                print("set INT8 mode.")
            
        # Parse model file
        print('Loading ONNX file from path {}...'.format(self.onnx_path))
        with open(self.onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print('Completed parsing of ONNX file')

        # check netowrk
        self._check_network(network)

        # build engine
        print('Building an engine from file {}; this may take a while...'.format(self.onnx_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")

        # save engine
        with open(self.plan_path, "wb") as f:
            f.write(plan)

        return engine

    def _get_engine(self):
        """generate tensorrt runtime engine
        """
        if os.path.exists(self.plan_path):
            print('Load trt plan from: {}'.format(self.plan_path))

            with open(self.plan_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            if os.path.exists(self.onnx_path):
                return self._parse_onnx()
            else:
                raise Exception("ONNX model file {} not exist!".format(self.onnx_path))

    def forward(self, image_tensors):
        """do infernece
        :param onnx_path: list, inputs tensor of model.
        :return outputs: list, outputs tensor of model.
        """

        for i, image_tensor in enumerate(image_tensors):
            image = np.array([image_tensor], dtype=np.float32, order='C')
            self.inputs[i].host = image

        trt_outputs = common.do_inference_v2(self.execution_context, 
                                             bindings=self.bindings,
                                             inputs=self.inputs,
                                             outputs=self.outputs,
                                             stream=self.stream)

        return trt_outputs
