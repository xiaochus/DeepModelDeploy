import os
import cv2
import torch
import numpy as np

from torchvision.models import resnet18


def export_pytorch_model(size, export_path, t='trace'):
    """TorchScript Convert
    """
    net = resnet18(pretrained=False)
    net.eval()

    if t == 'trace':
        # for no if/for control flow in net
        trace_model = torch.jit.trace(net, torch.Tensor(1, 3, size[0], size[1]))
        trace_model.save(export_path)
    else:
        script_model = torch.jit.script(net, torch.Tensor(1, 3, size[0], size[1]))
        script_model.save(export_path)


def export_onnx_model(size, export_path):
    """ONNX Convert
    """
    net = resnet18(pretrained=False)
    net.eval()

    input_names = ["input"]
    output_names = ["output"]
    batch = 1
    opset_version = 11
    dynamic = False
    if batch <= 0:
        dynamic = True
    
    if dynamic:
        dummy_input = torch.randn(1, 3, size[0], size[1]).to(device='cpu')
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        torch.onnx.export(net, dummy_input, export_path, 
                        verbose=False,
                        export_params=True,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes
                        )
    else:
        dummy_input = torch.randn(batch, 3, size[0], size[1]).to(device='cpu')

        torch.onnx.export(net, dummy_input, export_path, 
                        verbose=False,
                        export_params=True,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=None
                        )


def process_image(image_path, size, mean=[0, 0, 0], std=[1, 1, 1]):
    """process image
    """
    mean = np.array(mean)
    std = np.array(std)

    if not os.path.exists(image_path):
        raise Exception("image {} not exist!".format(image_path))
    img = cv2.imread(image_path)

    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img - mean
    img = img / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img
