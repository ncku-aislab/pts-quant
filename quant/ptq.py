import os
import sys
import argparse
from pathlib import Path

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
    mode: NotRequired[str]        # reconstruct or evaluate
    weight_path: NotRequired[str] # quantized checkpoint path

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
        raise ValueError("save_path must be provided when saving a calibrated model.")

    path = Path(save_path)

    # If save_path is a directory, save as {save_name}_s{s_iter}.pth
    if path.is_dir() or str(path).endswith("/"):
        return str(path / f"{save_name}_s{s_iter}.pth")

    # If multiple scale_iter values are used, avoid overwriting the same file
    if num_scale_iters > 1:
        return str(path.with_name(f"{path.stem}_s{s_iter}{path.suffix or '.pth'}"))

    return save_path


def save_quantized_checkpoint(
    model: nn.Module,
    config: ExperimentConfig,
    s_iter: int,
    save_path: str,
) -> None:
    if save_path is None:
        raise ValueError("save_path must be provided when saving a calibrated model.")

    final_save_path = resolve_save_path(
        save_path=save_path,
        save_name=config["save_name"],
        s_iter=s_iter,
        num_scale_iters=len(config["scale_iter"]),
    )

    Path(final_save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint_config = dict(config)
    checkpoint_config.update({
        "w_bits": config["wq_params"]["n_bits"],
        "a_bits": config["aq_params"]["n_bits"],
        "s_iter": s_iter,
    })

    checkpoint = {
        "model_name": config["model_name"],
        "save_name": config["save_name"],
        "state_dict": model.state_dict(),
        "quantizer_state": collect_quantizer_state(model),
        "quantized_weights": collect_quantized_weights(model),
        "config": checkpoint_config,
        "s_iter": s_iter,
    }

    torch.save(checkpoint, final_save_path)
    print(f"Quantized checkpoint saved to {final_save_path}")







def _get_checkpoint_metadata(weight_path: str) -> dict:
    """Load experiment metadata saved inside a quantized checkpoint.

    New checkpoints saved by save_quantized_checkpoint contain both `config` and
    `s_iter`. Older checkpoints may only contain a plain state_dict; in that
    case, return an empty dict so evaluation results do not incorrectly reuse
    the evaluation YAML settings.
    """
    checkpoint = torch.load(weight_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        return {}

    config = checkpoint.get("config", {})
    metadata = dict(config) if isinstance(config, dict) else {}

    if "s_iter" in checkpoint:
        metadata["s_iter"] = int(checkpoint["s_iter"])
    if "model_name" in checkpoint:
        metadata.setdefault("model_name", checkpoint["model_name"])
    if "save_name" in checkpoint:
        metadata.setdefault("save_name", checkpoint["save_name"])

    return metadata


def _get_config_value(metadata: dict, key: str, default="unknown"):
    value = metadata.get(key, default)
    return default if value is None else value


def evaluate_checkpoint(config: ExperimentConfig, device=None):
    """Evaluate an already reconstructed quantized checkpoint.

    This function is intentionally separated from calibrate() because evaluation
    is a different execution path from calibration/reconstruction.
    """
    model_name = config["model_name"]
    save_name = config["save_name"]
    wq_params = config["wq_params"]
    aq_params = config["aq_params"]
    result_path = config.get("result_path", "result_csv/ImageNet.csv")
    weight_path = config.get("weight_path", None)

    if weight_path is None:
        raise ValueError("weight_path must be provided when mode='evaluate'.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _, testloader = build_imagenet_data(
        data_path="data/ImageNet-1k/ILSVRC/Data/CLS-LOC",
        batch_size=16,
    )

    checkpoint_metadata = _get_checkpoint_metadata(weight_path)

    qnn = load_model(
        model_type="quantized",
        model_name=model_name,
        weight_path=weight_path,
        wq_params=wq_params,
        aq_params=aq_params,
    )

    qnn.to(device)
    qnn.eval()

    print(qnn)

    wq_params_from_ckpt = checkpoint_metadata.get("wq_params", {})
    aq_params_from_ckpt = checkpoint_metadata.get("aq_params", {})
    if not isinstance(wq_params_from_ckpt, dict):
        wq_params_from_ckpt = {}
    if not isinstance(aq_params_from_ckpt, dict):
        aq_params_from_ckpt = {}

    res = validate_model(testloader, qnn, device)
    res.update({
        "model": _get_config_value(checkpoint_metadata, "save_name", save_name),
        "mode": "evaluate",
        "weight_path": weight_path,
        "init_fn": _get_config_value(checkpoint_metadata, "initialization_fn"),
        "constraint_fn": _get_config_value(checkpoint_metadata, "constraint_fn"),
        "s_iter": _get_config_value(checkpoint_metadata, "s_iter"),
        "w_bits": _get_config_value(
            checkpoint_metadata,
            "w_bits",
            wq_params_from_ckpt.get("n_bits", "unknown"),
        ),
        "a_bits": _get_config_value(
            checkpoint_metadata,
            "a_bits",
            aq_params_from_ckpt.get("n_bits", "unknown"),
        ),
        "joint_training": _get_config_value(checkpoint_metadata, "joint_training"),
    })

    df = pd.DataFrame([res])
    df = save_csv(df, result_path, verbose=False)
    print(df)

    del qnn, testloader
    torch.cuda.empty_cache()


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
    mode = config.get("mode", "reconstruction")
    
    # Hyperparameters
    num_samples = 1024  #size of the calibration dataset
    iters_w = 20000      #number of iteration for adaround
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

    if mode not in ["reconstruction", "calibrate"]:
        raise ValueError(f"Unsupported mode for calibrate(): {mode}")

    # Dataset
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
            "mode": mode,
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
        
        # Save model weights
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
    



def run_experiment(config: ExperimentConfig, device=None):
    mode = config.get("mode", "reconstruction")

    if mode == "evaluate":
        return evaluate_checkpoint(config, device=device)
    if mode in ["reconstruction", "calibrate"]:
        return calibrate(config, device=device)

    raise ValueError(f"Unsupported mode: {mode}")


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
        run_experiment(model_config)