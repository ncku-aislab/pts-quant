import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, signed: bool = True, channel_wise: bool = False,
                 scale_method: str = 'minmax',
                 leaf_param: bool = False, prob: float = 1.0,
                 qparam_shape=None):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        self.signed = signed

        if self.sym and not self.signed:
            # You *can* implement unsigned symmetric, but it is uncommon and often confusing.
            raise ValueError("Symmetric quantization is typically used with signed integers (signed=True).")
        assert 2 <= n_bits <= 9, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.inited = True
        '''if leaf_param, use EMA to set scale'''
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.register_buffer("eps", torch.tensor(1e-8))

        if self.sym and not self.signed:
            # You *can* implement unsigned symmetric, but it is uncommon and often confusing.
            raise ValueError("Symmetric quantization is typically used with signed integers (signed=True).")

        if qparam_shape == None:
            #self.scale = nn.Parameter(torch.tensor(float(1.0)))
            self.register_buffer("delta", torch.tensor(float(1.0)))
            self.register_buffer("zero_point", torch.tensor(int(0)))
        else:
            #self.scale = nn.Parameter(torch.ones(qparam_shape, dtype=torch.float))
            self.register_buffer("delta", torch.ones(qparam_shape, dtype=torch.float))
            self.register_buffer("zero_point", torch.zeros(qparam_shape))

        '''mse params'''
        self.scale_method = 'mse'
        self.one_side_dist = None
        self.num = 100

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited
    
    def get_qrange(self):
        """
        Return integer quantization range [quant_min, quant_max].
        """
        if self.sym:
            if not self.signed:
                # Not supported in this version
                raise ValueError("Unsigned symmetric is not supported.")
            quant_min = -(2 ** (self.n_bits - 1))
            quant_max = (2 ** (self.n_bits - 1)) - 1
        else:
            quant_min = 0
            quant_max = self.n_levels - 1
        return quant_min, quant_max

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
        
        quant_min, quant_max = self.get_qrange()

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, quant_min, quant_max)
        x_dequant = (x_quant - self.zero_point) * self.delta
        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = self.get_qrange()

        if self.sym:
            # Symmetric: zp = 0, scale from max absolute
            max_abs = torch.max(max_val.abs(), min_val.abs())
            scale = max_abs / float(quant_max)  # quant_max is positive
            scale = torch.max(scale, self.eps)
            zero_point = torch.zeros_like(scale)
            return scale, zero_point

        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        quant_min, quant_max = self.get_qrange()
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta) + zero_point
        x_quant = torch.clamp(x_int, quant_min, quant_max)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def perform_2D_search(self, x):
        if self.sym:
            # symmetric -> no zp enumeration needed; fall back to 1D search
            return self.perform_1D_search(x)
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            # enumerate zp
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        if self.sym:
            xrange = torch.max(x_min.abs(), x_max.abs())
        else:
            xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            if self.sym:
                new_min = -thres
                new_max = thres
            else:
                new_min = torch.zeros_like(x_min) if self.one_side_dist == "pos" else -thres
                new_max = torch.zeros_like(x_max) if self.one_side_dist == "neg" else thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        if self.scale_method != 'mse':
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'

        if self.sym:
            best_min, best_max = self.perform_1D_search(x)
        else:
            if self.one_side_dist != "no":
                best_min, best_max = self.perform_1D_search(x)
            else:
                best_min, best_max = self.perform_2D_search(x)

        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            # determine the scale and zero point channel-by-channel
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def round_scale_to_pow2(self):
        with torch.no_grad():
            log2_scale = torch.log2(self.delta)
            rounded_log2 = torch.round(log2_scale)
            self.delta.copy_(torch.pow(2, rounded_log2))

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 9, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, is_training={}, inited={}'.format(
            self.n_bits, self.is_training, self.inited
        )


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant=False):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
            # in_channels and out_channels
            self.in_channels = org_module.in_channels
            self.out_channels = org_module.out_channels
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            # in_channels and out_channels
            self.in_channels = org_module.in_features
            self.out_channels = org_module.out_features
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        if weight_quant_params["channel_wise"]:
            if isinstance(org_module, nn.Conv2d):
                self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, qparam_shape=[self.out_channels, 1, 1, 1])
            elif isinstance(org_module, nn.Linear):
                self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, qparam_shape=[self.out_channels, 1])
        else:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.norm_function = StraightThrough()
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant
        self.trained = False

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        out = self.norm_function(out)
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    
    def adjust_quantized_model_scales(self):
        self.weight_quantizer.round_scale_to_pow2()
        self.act_quantizer.round_scale_to_pow2()

    @torch.jit.export
    def extra_repr(self):
        return 'wbit={}, abit={}, disable_act_quant={}'.format(
            self.weight_quantizer.n_bits, self.act_quantizer.n_bits, self.disable_act_quant
        )

class PTSQuantizer(nn.Module):
    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor = None,
                round_mode='learned_hard_sigmoid', pts_mode='learned_hard_sigmoid', constraint_fn='sigmoid'):
        super(PTSQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.signed = uaq.signed
        self.delta = uaq.delta
        self.log2_delta_floor = None
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.leaf_param = uaq.leaf_param #Check if quantizer is for weight or activation
        self.is_training = uaq.is_training
        self.prob = uaq.prob

        assert constraint_fn in ['sigmoid', 'tanh'], "constraint_fn must be 'sigmoid' or 'tanh'"
        self.weights_constraint_fn = 'sigmoid'
        self.scale_constraint_fn = constraint_fn

        # Adaround quantizer parameters (if quantizer for weight)
        if self.leaf_param == False:
            self.round_mode = round_mode
            self.alpha = None
            self.soft_targets = False
            # params for sigmoid function
            self.gamma, self.zeta = -0.1, 1.1
            self.beta = 2/3
            self.init_alpha(x=weight_tensor.clone())

        # PTS quantizer parameters
        self.pts_mode = pts_mode
        self.pts_alpha = None
        self.pts_soft_targets = False
        # params for sigmoid function
        self.pts_gamma, self.pts_zeta = -0.1, 1.1
        self.pts_beta = 2/3
        self.init_pts_alpha()
    
    def get_qrange(self):
        if self.sym:
            qmin = -(2 ** (self.n_bits - 1))
            qmax = (2 ** (self.n_bits - 1)) - 1
        else:
            qmin = 0
            qmax = self.n_levels - 1
        return qmin, qmax
    
    def forward(self, x):
        if self.pts_mode == 'learned_hard_sigmoid':
            if self.pts_soft_targets:
                soft = self.get_pts_soft_targets()
            else:
                soft = (self.pts_alpha >= 0).float()
            log2_delta = self.log2_delta_floor + soft
        elif self.pts_mode == 'normal':
            log2_delta = torch.log2(self.delta)
        else:
            raise NotImplementedError
        delta = torch.pow(2.0, log2_delta)

        qmin, qmax = self.get_qrange()

        # If quantizer is for weight
        if self.leaf_param == False:
            if self.round_mode == 'learned_hard_sigmoid':
                x_floor = torch.floor(x / delta)
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    x_int = x_floor + (self.alpha >= 0).float()
            else:
                raise ValueError('Wrong rounding mode')
            
            x_quant = torch.clamp(x_int + self.zero_point, qmin, qmax)
            x_float_q = (x_quant - self.zero_point) * delta

            return x_float_q
        # If quantizer is for activation
        else:   
            x_int = round_ste(x / delta) + self.zero_point
            x_quant = torch.clamp(x_int, qmin, qmax)
            x_dequant = (x_quant - self.zero_point) * delta
            if self.is_training and self.prob < 1.0:
                x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
            else:
                x_ans = x_dequant
            return x_ans

    
    def apply_constraint(self, alpha, gamma, zeta):
        """
        Map the unconstrained variable alpha to the interval [gamma, zeta]
        using a sigmoid- or tanh-based constraint function.
        """
        if self.weights_constraint_fn == 'sigmoid':
            out = torch.sigmoid(alpha)
        elif self.weights_constraint_fn == 'tanh':
            out = 0.5 * (torch.tanh(alpha) + 1)
        else:
            raise ValueError("Invalid constraint_fn")
        out = out * (zeta - gamma) + gamma
        return torch.clamp(out, 0, 1)

    def inverse_constraint(self, rest):
        """
        Compute the inverse mapping from a constrained variable in (0, 1)
        back to the corresponding unconstrained alpha.
        """
        eps = 1e-6
        rest = rest.clamp(eps, 1 - eps)
        if self.scale_constraint_fn == 'sigmoid':
            # inverse sigmoid
            return -torch.log((1 / rest) - 1)
        elif self.scale_constraint_fn == 'tanh':
            # inverse tanh mapping (torch.atanh)
            return torch.atanh(2 * rest - 1)
        else:
            raise ValueError("Invalid constraint_fn")
    
    def get_soft_targets(self):
        return self.apply_constraint(self.alpha, self.gamma, self.zeta)

    def get_pts_soft_targets(self):
        return self.apply_constraint(self.pts_alpha, self.pts_gamma, self.pts_zeta)
    
    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = self.inverse_constraint(rest)
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError
            
    def init_pts_alpha(self):
        if self.pts_mode == 'learned_hard_sigmoid':
            log2_scale = torch.log2(self.delta)
            s_floor = torch.floor(log2_scale)
            print('Init pts_alpha to be FP32')
            rest = log2_scale - s_floor  # rest of rounding [0, 1)
            pts_alpha = self.inverse_constraint(rest)
            self.pts_alpha = nn.Parameter(pts_alpha)
            self.log2_delta_floor = s_floor
        elif self.pts_mode == 'normal':
            pass
        else:
            raise NotImplementedError
    
    def convert_scale(self):
        soft = (self.pts_alpha >= 0).float()
        log2_delta = self.log2_delta_floor + soft
        self.delta = torch.pow(2.0, log2_delta)

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, scale={}'.format(self.n_bits, self.delta)
