import torch

from models.ResNet import resnet18 as _resnet18
from models.ResNet import resnet50 as _resnet50
from models.MobileNetV2 import mobilenetv2 as _mobilenetv2
from models.regnet import regnetx_600m as _regnetx_600m
from models.regnet import regnetx_3200m as _regnetx_3200m
from torch.hub import load_state_dict_from_url

from quant import QuantModel

def load_model(
        model_type:str,
        model_name:str,
        weight_path:str = None,
        wq_params = None,
        aq_params = None,
        pretrained = True, 
        **kwargs,
        ):
    if model_name == "ResNet18":
        # Call the model, load pretrained weights
        model = _resnet18(**kwargs)
        if pretrained:
            load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar'
            checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
            model.load_state_dict(checkpoint)
    
    elif model_name == "ResNet50":
            # Call the model, load pretrained weights
        model = _resnet50(**kwargs)
        if pretrained:
            load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet50_imagenet.pth.tar'
            checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
            model.load_state_dict(checkpoint)
        return model
    
    elif model_name == "MobileNetV2":
        # Call the model, load pretrained weights
        model = _mobilenetv2(**kwargs)
        if pretrained:
            load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/mobilenetv2.pth.tar'
            checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
            model.load_state_dict(checkpoint['model'])
        return model
    elif model_name == "RegNetX600-MF":
        # Call the model, load pretrained weights
        model = _regnetx_600m(**kwargs)
        if pretrained:
            load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_600m.pth.tar'
            checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
            model.load_state_dict(checkpoint)
        return model
    elif model_name == "RegNetX3.2-GF":
        # Call the model, load pretrained weights
        model = _regnetx_3200m(**kwargs)
        if pretrained:
            load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_3200m.pth.tar'
            checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
            model.load_state_dict(checkpoint)
        return model
    else:
        raise NotImplementedError
    
    return model