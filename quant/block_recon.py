import torch
import torch.nn.functional as F
from .quant_layer import QuantModule, lp_loss, PTSQuantizer
from .quant_model import QuantModel
from .quant_block import BaseQuantBlock, specials_unquantized
from .set_weight_quantize_params import get_init, get_dc_fp_init
from .set_act_quantize_params import set_act_quantize_params

include = False

def is_power_of_two(t: torch.Tensor, tol=1e-6):
    """
    Check whether each element in the input tensor is an integer power of two.

    The function computes log2(t) and verifies that the result is sufficiently
    close to an integer value within a given tolerance.
    """
    log2_vals = torch.log2(t)
    return torch.all(torch.abs(log2_vals - torch.round(log2_vals)) < tol)

def find_unquantized_module(model: torch.nn.Module, module_list: list = [], name_list: list = []):
    """Store subsequent unquantized modules in a list"""
    global include
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                include = True
                module.set_quant_state(False,False)
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized:
            name_list.append(name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list)
    return module_list[1:], name_list[1:]

def block_reconstruction(model: QuantModel, fp_model: QuantModel, block: BaseQuantBlock, fp_block: BaseQuantBlock,
                        cali_data: torch.Tensor, batch_size: int = 16, iters: int = 20000, weight: float = 0.01, 
                        opt_mode: str = 'mse', b_range: tuple = (20, 2),
                        warmup: float = 0.2, p: float = 2.0, lr: float = 4e-5,
                        input_prob: float = 1.0, keep_gpu: bool = True, 
                        lamb_r: float = 0.2, T: float = 7.0, bn_lr: float = 1e-3, lamb_c=0.02,
                        scale_iter: int = 10000, constraint_fn: str = 'sigmoid', initialization_fn: str = 'sigmoid', joint_training: bool = True):
    """
    Reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param lr: not used now, used in PD-Quant for act scale learning
    :param p: L_p norm minimization
    :param lamb_r: hyper-parameter for regularization
    :param T: temperature coefficient for KL divergence
    :param bn_lr: learning rate for DC
    :param lamb_c: hyper-parameter for DC
    :param scale_iter: scale factor training iteration
    :param constraint_fn: constraint function for the rounding value, must be 'sigmoid' or 'tanh'
    :param initialization_fn: initialization function for alpha, must be 'sigmoid' or 'tanh' or 'zero' or 'random'
    :param joint_training: whether to train the weight and activation scale jointly, must be True or False
    """

    '''get input and set scale'''
    cached_inps = get_init(model, block, cali_data, batch_size=batch_size, 
                                        input_prob=True, keep_gpu=False).pin_memory()
    cached_outs, cached_output, cur_syms = get_dc_fp_init(fp_model, fp_block, cali_data, batch_size=batch_size, 
                                        input_prob=True, keep_gpu=False, bn_lr=bn_lr, lamb=lamb_c)
    cached_outs, cached_output, cur_syms = cached_outs.pin_memory(), cached_output.pin_memory(), cur_syms.pin_memory()
    set_act_quantize_params(block, cali_data=cached_inps[:min(256, cached_inps.size(0))])

    '''set state'''
    cur_weight, cur_act = True, True
    
    global include
    module_list, name_list, include = [], [], False
    module_list, name_list = find_unquantized_module(model, module_list, name_list)
    block.set_quant_state(cur_weight, cur_act)
    for para in model.parameters():
        para.requires_grad = False

    '''set quantizer'''
    round_mode = 'learned_hard_sigmoid'
    pts_mode = 'learned_hard_sigmoid'
    # Replace weight quantizer to PTSQuantizer
    w_para, ws_para, a_para = [], [], []
    w_opt, ws_opt, a_opt = None, None, None
    scheduler, ws_scheduler, a_scheduler = None, None, None

    for module in block.modules():
        '''weight'''
        if isinstance(module, QuantModule):
            module.weight_quantizer = PTSQuantizer(uaq=module.weight_quantizer, round_mode=round_mode, pts_mode=pts_mode,
                                               weight_tensor=module.org_weight.data, constraint_fn=constraint_fn, initialization_fn=initialization_fn)
            module.weight_quantizer.soft_targets = True
            w_para += [module.weight_quantizer.alpha]
            if pts_mode == 'learned_hard_sigmoid':
                ws_para += [module.weight_quantizer.pts_alpha]
                module.weight_quantizer.pts_soft_targets = True
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if module.act_quantizer.scale is not None:
                module.act_quantizer = PTSQuantizer(uaq=module.act_quantizer, round_mode=round_mode, pts_mode=pts_mode, constraint_fn=constraint_fn, initialization_fn=initialization_fn)
                if pts_mode == 'learned_hard_sigmoid':
                    a_para += [module.act_quantizer.pts_alpha]
                    module.act_quantizer.pts_soft_targets = True
            '''set up drop'''
            module.act_quantizer.is_training = True
    
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=3e-3)
    if len(ws_para) != 0:
        ws_opt = torch.optim.Adam(ws_para, lr=3e-3)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=3e-3)

    loss_mode = 'relaxation'
    rec_loss = opt_mode
    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p, lam=lamb_r, T=T)
    device = 'cuda'
    sz = cached_inps.size(0)
    for i in range(iters):
        if i == scale_iter:
            for module in block.modules():
                if isinstance(module, QuantModule):
                    module.weight_quantizer.convert_scale()
                    module.weight_quantizer.pts_mode = 'normal'
                    if not is_power_of_two(module.weight_quantizer.scale):
                        print('Warning: weight scale is not power of two')
                        break
                if isinstance(module, (QuantModule, BaseQuantBlock)):
                    if module.act_quantizer.scale is not None:
                        module.act_quantizer.convert_scale()
                        module.act_quantizer.pts_mode = 'normal'
                        if not is_power_of_two(module.act_quantizer.scale):
                            print('Warning: act scale is not power of two')
                            break
        idx = torch.randint(0, sz, (batch_size,))
        cur_inp = cached_inps[idx].cuda(non_blocking=True)
        cur_sym = cur_syms[idx].cuda(non_blocking=True)
        output_fp = cached_output[idx].cuda(non_blocking=True)
        cur_out = cached_outs[idx].cuda(non_blocking=True)
        if input_prob < 1.0:
            drop_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
        
        cur_inp = torch.cat((drop_inp, cur_inp))

        if w_opt:
            w_opt.zero_grad()
        if ws_opt:
            ws_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()
        
        out_all = block(cur_inp)

        '''forward for prediction difference'''
        out_drop = out_all[:batch_size]
        out_quant = out_all[batch_size:]
        output = out_quant
        for num, module in enumerate(module_list):
            # for ResNet and RegNet
            if name_list[num] == 'fc':
                output = torch.flatten(output, 1)
            # for MobileNet and MNasNet
            if isinstance(module, torch.nn.Dropout):
                output = output.mean([2, 3])
            output = module(output)
        err = loss_func(out_drop, cur_out, output, output_fp)

        err.backward(retain_graph=True)
        # Stage 1: i < scale_iter
        if i < scale_iter:
            if joint_training:
                # Joint training: update both weight rounding and scale rounding
                if w_opt:
                    w_opt.step()
                if ws_opt:
                    ws_opt.step()
                if a_opt:
                    a_opt.step()

                if scheduler:
                    scheduler.step()
                if ws_scheduler:
                    ws_scheduler.step()
                if a_scheduler:
                    a_scheduler.step()
            else:
                # Separate training: update only scale rounding
                if ws_opt:
                    ws_opt.step()
                if a_opt:
                    a_opt.step()

                if ws_scheduler:
                    ws_scheduler.step()
                if a_scheduler:
                    a_scheduler.step()

        # Stage 2: i >= scale_iter
        else:
            # Only update weight rounding
            if w_opt:
                w_opt.step()
            if scheduler:
                scheduler.step()
        
    torch.cuda.empty_cache()
    for module in block.modules():
        '''weight '''
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False
            module.weight_quantizer.pts_soft_targets = False
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.act_quantizer.pts_soft_targets = False
            module.act_quantizer.is_training = False
            module.trained = True
    for module in fp_block.modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.trained = True


class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 lam: float = 1.0,
                 T: float = 7.0):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.pd_loss = torch.nn.KLDivLoss(reduction='batchmean')

    def __call__(self, pred, tgt, output, output_fp):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy, pd_loss is the 
        prediction difference loss.

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param output: prediction from quantized model
        :param output_fp: prediction from FP model
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        pd_loss = self.pd_loss(F.log_softmax(output / self.T, dim=1), F.softmax(output_fp / self.T, dim=1)) / self.lam

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = 0
            round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss + pd_loss
        if self.count % 500 == 0:
            print(f'Total loss: {total_loss:.3f} (rec: {rec_loss:.3f}, pd: {pd_loss:.3f}, '
                f'round: {round_loss:.3f} ) '
                f'b={b:.2f}  count={self.count}')
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 20, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
