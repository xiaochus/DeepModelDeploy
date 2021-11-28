import os
import numpy as np
from rknn.api import RKNN


class RKNNMolde:
    def __init__(self, model_path, rknn_path, mode="fp16", device='rv1126', mean=[0, 0, 0], std=[1, 1, 1], quant_data_file="", pre_compile=False):
        """
        :param model_path: mode path, end with .cfg/.onnx/.pt/.tflite.
        :param rknn_path: save/read rknn model path.
        :param mode: model mode, fp16/uint8/int8/int16/hybrid.
        :param device: target chips, ie, 'rv1126', ['rv1126'].
        :param mean: mean operation for quant image, range(0-255).
        :param std: std operation for quant image, range(0-255).
        :param quant_data_file: file for quant image data.
        :param pre_compile: if pre-comiple model file for target device.
        """
        self.model_path = model_path
        self.rknn_path = rknn_path

        self.mode = mode
        self.device = device

        # quant param
        self.mean = mean
        self.std = std
        self.quant_data_file = quant_data_file
        self.pre_compile = pre_compile

        self.rknn = RKNN()

        self._get_model()

    def _export_model(self):
        """generate rknn model file.
        """
        if (self.model_path.endswith('.cfg')):
            model_type = "darknet"
        elif (self.model_path.endswith('onnx')):
            model_type = "onnx"
        elif (self.model_path.endswith('pt')):
            model_type = "pytorch"
        elif (self.model_path.endswith('tflite')):
            model_type = "tflite"
        else:
            model_type = ""
            raise Exception("Not support this model format!")

        if self.mode == "int8":
            quantized_dtype = "dynamic_fixed_point-i8"
        elif self.mode == "uint8":
            quantized_dtype = "asymmetric_quantized-u8"
        elif self.mode == "int16":
            quantized_dtype = "dynamic_fixed_point-i16"
        else:
            quantized_dtype = "asymmetric_quantized-u8"

        # config
        self.rknn.config(batch_size=1, 
                         mean_values=[self.mean], 
                         std_values=[self.std], 
                         reorder_channel='0 1 2',
                         target_platform=self.device,
                         quantized_dtype=quantized_dtype
                         )

        # parse model
        if model_type == "darknet":
            weight_file = self.model_path[:-4] + "weights"
            ret = self.rknn.load_darknet(model=self.model_path, weight=weight_file)
        elif model_type == "onnx":
            ret = self.rknn.load_onnx(model=self.model_path)
        elif model_type == "pytorch":
            ret = self.rknn.load_pytorch(model=self.model_path)
        elif model_type == "tflite":
            ret = self.rknn.load_tflite(model=self.model_path)
        else:
            ret = -1

        if ret != 0:
            raise Exception("Load model failed!")

        # quant
        print("set {} mode.".format(self.mode))

        if self.mode == "fp16":
            ret = self.rknn.build(do_quantization=False, dataset=self.quant_data_file, pre_compile=self.pre_compile)
        elif self.mode in ["uint8", "int8", "int16"]:
            ret = self.rknn.build(do_quantization=True, dataset=self.quant_data_file, pre_compile=self.pre_compile)
        elif self.mode == "hybrid":
            if model_type == "onnx":
                hybird_model_input = "torchjitexport.json"
                hybird_data_input = "torchjitexport.data"
                hybird_model_quantization_cfg = "torchjitexport.quantization.cfg"
            else:
                hybird_model_input = "rknn_hybrid_quant.json"
                hybird_data_input = "rknn_hybrid_quant.data"
                hybird_model_quantization_cfg = "rknn_hybrid_quant.cfg"

            if os.path.exists(hybird_model_input):
                os.remove(hybird_model_input)
            if os.path.exists(hybird_data_input):
                os.remove(hybird_data_input)
            if os.path.exists(hybird_model_quantization_cfg):
                os.remove(hybird_model_quantization_cfg)

            ret = self.rknn.hybrid_quantization_step1(dataset=self.quant_data_file)
            if ret != 0:
                raise Exception("Hybrid quant step1 failed!")

            ret = self.rknn.hybrid_quantization_step2(model_input=hybird_model_input,
                                                      data_input=hybird_data_input,
                                                      model_quantization_cfg=hybird_model_quantization_cfg,
                                                      dataset=self.quant_data_file,
                                                      pre_compile=self.pre_compile
                                                     )
            if ret != 0:
                raise Exception("Hybrid quant step2 failed!")
        else:
            ret = -1
            raise Exception("Not support mode {}!".format(self.mode))

        if ret != 0:
            raise Exception("Build rknn model failed!")

        # export model file
        ret = self.rknn.export_rknn(self.rknn_path)
        if ret != 0:
            raise Exception("Export rknn model failed!")
        
        print("Model build done.")

    def _get_model(self):
        """get rknn model
        """
        if os.path.exists(self.rknn_path):
             # load model
            ret = self.rknn.load_rknn(self.rknn_path)
            if ret != 0:
                raise Exception("Load rknn model failed!")

            # init runtime enviroment
            ret = self.rknn.init_runtime(target=None)
            if ret != 0:
                raise Exception("Init runtime enviroment failed!")
        else:
            if os.path.exists(self.model_path):
                self._export_model()

                # init runtime enviroment
                ret = self.rknn.init_runtime(target=None)
                if ret != 0:
                    raise Exception("Init runtime enviroment failed!")
            else:
                raise Exception("model file {} not exist!".format(self.model_path))

    def forward(self, image_tensors):
        """do infernece
        :param image_tensors: list, inputs of model.
        return outputs: list, outputs of model.
        """
        inputs = [image_tensor.astype(np.float32) for image_tensor in image_tensors]
        outputs = self.rknn.inference(inputs)

        return outputs

    def perf(self, image_tensors):
        """eval model performance on target device, pre_compile must be false.
        :param image_tensors: list, inputs of model.
        """
        inputs = [image_tensor.astype(np.float32) for image_tensor in image_tensors]
        pref_result = self.rknn.eval_perf(inputs, is_print=True)
        print(pref_result)
