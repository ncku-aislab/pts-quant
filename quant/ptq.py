import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import copy
import pandas as pd
from typing import TypedDict, NotRequired, List

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

    # Dataset
    trainloader, testloader = build_imagenet_data(data_path="data/ILSVRC2012", batch_size=16)
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