import torch
import torch.nn as nn
from typing import Optional, Dict

from models.ResNet import resnet18 as _resnet18
from models.ResNet import resnet50 as _resnet50
from models.MobileNetV2 import mobilenetv2 as _mobilenetv2
from models.regnet import regnetx_600m as _regnetx_600m
from models.regnet import regnetx_3200m as _regnetx_3200m
from torch.hub import load_state_dict_from_url

from quant import QuantModel, QuantModule, BaseQuantBlock, PTSQuantizer


def _build_fp_model(model_name: str, pretrained: bool = True, **kwargs):
    if model_name == "ResNet18":
        model = _resnet18(**kwargs)
        if pretrained:
            load_url = "https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar"
            checkpoint = load_state_dict_from_url(url=load_url, map_location="cpu", progress=True)
            model.load_state_dict(checkpoint)

    elif model_name == "ResNet50":
        model = _resnet50(**kwargs)
        if pretrained:
            load_url = "https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet50_imagenet.pth.tar"
            checkpoint = load_state_dict_from_url(url=load_url, map_location="cpu", progress=True)
            model.load_state_dict(checkpoint)

    elif model_name == "MobileNetV2":
        model = _mobilenetv2(**kwargs)
        if pretrained:
            load_url = "https://github.com/yhhhli/BRECQ/releases/download/v1.0/mobilenetv2.pth.tar"
            checkpoint = load_state_dict_from_url(url=load_url, map_location="cpu", progress=True)
            model.load_state_dict(checkpoint["model"])

    elif model_name == "RegNetX-600MF":
        model = _regnetx_600m(**kwargs)
        if pretrained:
            load_url = "https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_600m.pth.tar"
            checkpoint = load_state_dict_from_url(url=load_url, map_location="cpu", progress=True)
            model.load_state_dict(checkpoint)

    elif model_name == "RegNetX-3.2GF":
        model = _regnetx_3200m(**kwargs)
        if pretrained:
            load_url = "https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_3200m.pth.tar"
            checkpoint = load_state_dict_from_url(url=load_url, map_location="cpu", progress=True)
            model.load_state_dict(checkpoint)

    else:
        raise NotImplementedError(f"Unsupported model_name: {model_name}")

    return model


def _replace_quantizers_with_pts(model: nn.Module, quantizer_state: Optional[Dict[str, Dict]] = None):
    if quantizer_state is None:
        quantizer_state = {}

    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            wq_name = f"{name}.weight_quantizer"
            wq_state = quantizer_state.get(wq_name, {})

            module.weight_quantizer = PTSQuantizer(
                uaq=module.weight_quantizer,
                round_mode=wq_state.get("round_mode", "learned_hard_sigmoid"),
                # Important: use learned_hard_sigmoid while loading so pts_alpha exists
                pts_mode="learned_hard_sigmoid",
                weight_tensor=module.org_weight.data,
                constraint_fn=wq_state.get("constraint_fn", "sigmoid"),
                initialization_fn=wq_state.get("initialization_fn", "sigmoid"),
            )

            aq_name = f"{name}.act_quantizer"
            aq_state = quantizer_state.get(aq_name, {})

            if module.act_quantizer.scale is not None:
                module.act_quantizer = PTSQuantizer(
                    uaq=module.act_quantizer,
                    round_mode=wq_state.get("round_mode", "learned_hard_sigmoid"),
                    # Important: use learned_hard_sigmoid while loading so pts_alpha exists
                    pts_mode="learned_hard_sigmoid",
                    constraint_fn=aq_state.get("constraint_fn", "sigmoid"),
                    initialization_fn=aq_state.get("initialization_fn", "sigmoid"),
                )

        elif isinstance(module, BaseQuantBlock):
            aq_name = f"{name}.act_quantizer"
            aq_state = quantizer_state.get(aq_name, {})

            if hasattr(module, "act_quantizer") and module.act_quantizer.scale is not None:
                module.act_quantizer = PTSQuantizer(
                    uaq=module.act_quantizer,
                    pts_mode="learned_hard_sigmoid",
                    constraint_fn=aq_state.get("constraint_fn", "sigmoid"),
                    initialization_fn=aq_state.get("initialization_fn", "sigmoid"),
                )



def _load_quantized_model(
    model_name: str,
    checkpoint: Optional[dict],
    wq_params: dict,
    aq_params: dict,
    **kwargs,
):
    if checkpoint is None:
        raise ValueError("weight_path must be provided when loading a quantized model.")
    if wq_params is None or aq_params is None:
        raise ValueError("wq_params and aq_params must be provided when loading a quantized model.")

    base_model = _build_fp_model(model_name, pretrained=False, **kwargs)
    model = QuantModel(
        model=base_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
    )
    # Must match the reconstruction-time quantization settings
    model.set_first_last_layer_to_8bit()
    model.disable_network_output_quantization()

    quantizer_state = checkpoint.get("quantizer_state", {})
    _replace_quantizers_with_pts(model, quantizer_state)

    state_dict = checkpoint.get("state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    model.set_quant_state(weight_quant=True, act_quant=True)
    model.eval()

    print(f"Loaded quantized model from checkpoint")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    

    model.set_quant_state(weight_quant=True, act_quant=True)

    for module in model.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.pts_mode = "normal"
            module.weight_quantizer.soft_targets = False
            module.weight_quantizer.pts_soft_targets = False

            module.act_quantizer.pts_mode = "normal"
            module.act_quantizer.pts_soft_targets = False
            module.act_quantizer.is_training = False

        elif isinstance(module, BaseQuantBlock):
            module.act_quantizer.pts_mode = "normal"
            module.act_quantizer.pts_soft_targets = False
            module.act_quantizer.is_training = False

    model.eval()

    return model


def load_model(
    model_type: str,
    model_name: str,
    checkpoint: Optional[dict]=None,
    wq_params=None,
    aq_params=None,
    pretrained: bool = True,
    **kwargs,
):
    if model_type == "full":
        return _build_fp_model(model_name, pretrained=pretrained, **kwargs)

    elif model_type in ["quant", "quantized"]:
        return _load_quantized_model(
            model_name=model_name,
            checkpoint=checkpoint,
            wq_params=wq_params,
            aq_params=aq_params,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")