import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import copy
import pandas as pd
from typing import TypedDict, List
from typing_extensions import NotRequired

from utils import *
from quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)

class QuantParams(TypedDict):
    n_bits: int
    symmetric: bool
    channel_wise: bool
    scale_method: str
    leaf_param: NotRequired[bool]
    prob: NotRequired[float]


class ExperimentConfig(TypedDict):
    model_name: str
    save_name: str
    wq_params: QuantParams
    aq_params: QuantParams
    constraint_fn: str
    initialization_fn: str
    scale_iter: List[int]
    joint_training: bool
    result_path: NotRequired[str]
    save_path: NotRequired[str]

def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def collect_quantizer_state(model: nn.Module) -> dict:
    quantizer_state = {}

    def add_quantizer_state(name: str, quantizer):
        state = {
            "n_bits": getattr(quantizer, "n_bits", None),
            "scale": _to_cpu(getattr(quantizer, "scale", None)),
            "zero_point": _to_cpu(getattr(quantizer, "zero_point", None)),
            "alpha": _to_cpu(getattr(quantizer, "alpha", None)),
            "pts_alpha": _to_cpu(getattr(quantizer, "pts_alpha", None)),
            "log2_scale_floor": _to_cpu(getattr(quantizer, "log2_scale_floor", None)),
            "round_mode": getattr(quantizer, "round_mode", None),
            "pts_mode": getattr(quantizer, "pts_mode", None),
            "constraint_fn": getattr(quantizer, "constraint_fn", None),
            "initialization_fn": getattr(quantizer, "initialization_fn", None),
        }
        quantizer_state[name] = state

    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            add_quantizer_state(f"{name}.weight_quantizer", module.weight_quantizer)
            add_quantizer_state(f"{name}.act_quantizer", module.act_quantizer)

        elif isinstance(module, BaseQuantBlock):
            if hasattr(module, "act_quantizer"):
                add_quantizer_state(f"{name}.act_quantizer", module.act_quantizer)

    return quantizer_state


def collect_quantized_weights(model: nn.Module) -> dict:
    quantized_weights = {}

    for name, module in model.named_modules():
        if not isinstance(module, QuantModule):
            continue

        quantizer = module.weight_quantizer
        weight = module.weight.detach()

        with torch.no_grad():
            qmin, qmax = quantizer.get_qrange()
            scale = quantizer.scale
            zero_point = quantizer.zero_point

            if hasattr(quantizer, "alpha") and quantizer.alpha is not None:
                weight_floor = torch.floor(weight / scale)
                weight_int = weight_floor + (quantizer.alpha >= 0).float()
            else:
                weight_int = torch.round(weight / scale)

            weight_int = torch.clamp(weight_int + zero_point, qmin, qmax)
            weight_dequant = (weight_int - zero_point) * scale

        quantized_weights[name] = {
            "int_weight": weight_int.detach().cpu(),
            "dequant_weight": weight_dequant.detach().cpu(),
        }

    return quantized_weights


def resolve_save_path(save_path: str, save_name: str, s_iter: int, num_scale_iters: int) -> str:
    if save_path is None:
        return None

    # If save_path is a directory, save as {save_name}_s{s_iter}.pth
    if save_path.endswith("/") or os.path.isdir(save_path):
        return os.path.join(save_path, f"{save_name}_s{s_iter}.pth")

    # If multiple scale_iter values are used, avoid overwriting the same file
    if num_scale_iters > 1:
        base, ext = os.path.splitext(save_path)
        ext = ext if ext else ".pth"
        return f"{base}_s{s_iter}{ext}"

    return save_path


def save_quantized_checkpoint(
    model: nn.Module,
    config: ExperimentConfig,
    s_iter: int,
    save_path: str,
) -> None:
    if save_path is None:
        return

    final_save_path = resolve_save_path(
        save_path=save_path,
        save_name=config["save_name"],
        s_iter=s_iter,
        num_scale_iters=len(config["scale_iter"]),
    )

    save_dir = os.path.dirname(final_save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "model_name": config["model_name"],
        "save_name": config["save_name"],
        "state_dict": model.state_dict(),
        "quantizer_state": collect_quantizer_state(model),
        "quantized_weights": collect_quantized_weights(model),
        "config": dict(config),
        "s_iter": s_iter,
    }

    torch.save(checkpoint, final_save_path)
    print(f"Quantized checkpoint saved to {final_save_path}")




def calibrate(config: ExperimentConfig, device=None):
    # Read config
    model_name = config["model_name"]
    save_name = config["save_name"]
    wq_params = config["wq_params"]
    aq_params = config["aq_params"]
    constraint_fn = config["constraint_fn"]
    initialization_fn = config["initialization_fn"]
    scale_grid = config["scale_iter"]
    joint_training = config["joint_training"]
    result_path = config.get("result_path", "result_csv/ImageNet.csv")
    save_path = config.get("save_path", None)
    
    # Hyperparameters
    num_samples = 1024  #size of the calibration dataset
    iters_w = 2      #number of iteration for adaround
    batch_size = 16     #number of batch size
    weight = 0.01       #weight of rounding cost vs the reconstruction loss

    b_start = 20        #temperature at the beginning of calibration
    b_end = 2           #temperature at the end of calibration
    warmup = 0.2        #in the warmup period no regularization is applied

    lr = 4e-5           #learning rate for LSQ

    lamb_r = 0.1        #hyper-parameter for regularization
    Temp = 4.0          #temperature coefficient for KL divergence
    bn_lr = 1e-3        #learning rate for DC
    lamb_c = 0.02       #hyper-parameter for DC

    # Dataset
    #trainloader, testloader = build_imagenet_data(data_path="data/ILSVRC2012", batch_size=16)
    trainloader, testloader = build_imagenet_data(data_path="data/ImageNet-1k/ILSVRC/Data/CLS-LOC", batch_size=16)
    trainloader, calibloader = split_data(trainloader, num_samples)
    cali_data, _ = get_train_samples(calibloader, num_samples)

    #model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("full", model_name)
    model.cuda()
    model.eval()

    print(model)

    # FP model
    fp_model = copy.deepcopy(model)
    fp_model.cuda()
    fp_model.eval()

    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.cuda()
    fp_model.eval()
    fp_model.set_quant_state(False, False)

    kwargs = dict(cali_data=cali_data, batch_size=batch_size, iters=iters_w, weight=weight,
                b_range=(b_start, b_end), warmup=warmup, opt_mode='mse',
                lr=lr, input_prob=0.5, keep_gpu=True, 
                lamb_r=lamb_r, T=Temp, bn_lr=bn_lr, lamb_c=lamb_c, scale_iter=scale_grid[0], 
                constraint_fn=constraint_fn, initialization_fn=initialization_fn, joint_training=joint_training)
    
    def set_weight_act_quantize_params(module, fp_module):
        if isinstance(module, QuantModule):
            layer_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            block_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        else:
            raise NotImplementedError
    def recon_model(model: nn.Module, fp_model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for (name, module), (_, fp_module) in zip(model.named_children(), fp_model.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            elif isinstance(module, BaseQuantBlock):
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            else:
                recon_model(module, fp_module)
    # Start calibration
    for s_iter in scale_grid:
        kwargs['scale_iter'] = s_iter
        qnn = QuantModel(copy.deepcopy(model), weight_quant_params=wq_params, act_quant_params=aq_params)
        qnn.cuda()
        qnn.eval()

        qnn.set_first_last_layer_to_8bit()
        qnn.disable_network_output_quantization()
        print('the quantized model is below!')
        print(qnn)
        # init weight quantizer
        set_weight_quantize_params(qnn)

        # Calibration
        recon_model(qnn, fp_model)
        
        # Set the quant model to the inference mode
        qnn.set_quant_state(weight_quant=True, act_quant=True)

        for module in qnn.modules():
            if isinstance(module, QuantModule):
                module.weight_quantizer.convert_scale()
                module.weight_quantizer.pts_mode = 'normal'
                module.act_quantizer.convert_scale()
                module.act_quantizer.pts_mode = 'normal'
            elif isinstance(module, BaseQuantBlock):
                module.act_quantizer.convert_scale()
                module.act_quantizer.pts_mode = 'normal'

        res = validate_model(testloader, qnn, device)
        res.update({
            "model": save_name,
            "init_fn": initialization_fn,
            "constraint_fn": constraint_fn,
            "s_iter": s_iter,
            "w_bits": wq_params["n_bits"],
            "a_bits": aq_params["n_bits"],
            "joint_training": joint_training,
        })

        df = pd.DataFrame([res])
        df = save_csv(df, result_path, verbose=False)

        print(df)

        save_quantized_checkpoint(
            model=qnn,
            config=config,
            s_iter=s_iter,
            save_path=save_path,
        )


    
    
    # clear memory
    del model, fp_model, qnn, cali_data, trainloader, testloader
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTQ calibration script")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml file"
    )

    args_cli = parser.parse_args()

    # load yaml config
    args = load_config(args_cli.config)

    # run calibration
    for model_config in args.models:
        calibrate(model_config)