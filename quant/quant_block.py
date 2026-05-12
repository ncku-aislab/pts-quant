import torch.nn as nn
from .quant_layer import QuantModule, UniformAffineQuantizer
from models.imagenet.ResNet import BasicBlock_imagenet, Bottleneck_imagenet
from models.imagenet.MobileNetV2 import InvertedResidual_imagenet
from models.imagenet.regnet import ResBottleneckBlock_imagenet

from models.cifar10.ResNet import BasicBlock_cifar10, Bottleneck_cifar10
from models.cifar10.MobileNetV2 import Block
from models.cifar10.regnet import ResBottleneckBlock_cifar10



class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.ignore_reconstruction = False
        self.trained = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock_imagenet(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock_imagenet used in ResNet-18.
    """
    def __init__(self, basic_block: BasicBlock_imagenet, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = basic_block.bn1
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv2.norm_function = basic_block.bn2

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = basic_block.downsample[1]
        self.activation_function = basic_block.relu2
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck_imagenet(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck_imagenet Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck_imagenet, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = bottleneck.bn1
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.norm_function = bottleneck.bn2
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = bottleneck.bn3

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = bottleneck.downsample[1]
        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual_imagenet(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual_imagenet, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].norm_function = inv_res.conv[1]
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].norm_function = inv_res.conv[4]
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].norm_function = inv_res.conv[1]
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].norm_function = inv_res.conv[4]
            self.conv[1].activation_function = nn.ReLU6()
            self.conv[2].norm_function = inv_res.conv[7]
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out
    
class QuantResBottleneckBlock_imagenet(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock_imagenet, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.norm_function = bottleneck.f.a_bn
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.norm_function = bottleneck.f.b_bn
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = bottleneck.f.c_bn

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = bottleneck.bn
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

        self.activation_function = bottleneck.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBasicBlock_cifar10(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock_cifar10 used in ResNet-18.
    """
    def __init__(self, basic_block: BasicBlock_cifar10, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = basic_block.bn1
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv2.norm_function = basic_block.bn2

        if len(basic_block.shortcut) == 0:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.shortcut[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = basic_block.shortcut[1]
        self.activation_function = basic_block.relu2
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

class QuantBottleneck_cifar10(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck_cifar10 Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck_cifar10, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = bottleneck.bn1
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.norm_function = bottleneck.bn2
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = bottleneck.bn3

        if len(bottleneck.shortcut) == 0:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.shortcut[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
            self.downsample.norm_function = bottleneck.shortcut[1]
        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

class QuantInvertedResidual_cifar10(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: Block, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        #
        self.stride = inv_res.stride

        self.conv1 = QuantModule(inv_res.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = inv_res.bn1
        self.conv1.activation_function = inv_res.relu1

        self.conv2 = QuantModule(inv_res.conv2, weight_quant_params, act_quant_params)
        self.conv2.norm_function = inv_res.bn2
        self.conv2.activation_function = inv_res.relu2

        self.conv3 = QuantModule(inv_res.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = inv_res.bn3

        self.shortcut = nn.Sequential()
        if len(inv_res.shortcut) != 0:
            self.shortcut = QuantModule(inv_res.shortcut[0], weight_quant_params, act_quant_params, disable_act_quant=True)
            self.shortcut.norm_function = inv_res.shortcut[1]
    
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class QuantResBottleneckBlock_cifar10(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock_cifar10, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = bottleneck.bn1
        self.conv1.activation_function = bottleneck.relu1

        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.norm_function = bottleneck.bn2
        self.conv2.activation_function = bottleneck.relu2

        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = bottleneck.bn3

        if len(bottleneck.shortcut) == 0:
            self.shortcut = None
        else:
            self.shortcut = QuantModule(bottleneck.shortcut[0], weight_quant_params, act_quant_params, disable_act_quant=True)
            self.shortcut.norm_function = bottleneck.shortcut[1]
        
        self.activation_function = bottleneck.relu3
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


specials = {
    BasicBlock_imagenet: QuantBasicBlock_imagenet,
    Bottleneck_imagenet: QuantBottleneck_imagenet,
    InvertedResidual_imagenet: QuantInvertedResidual_imagenet,
    ResBottleneckBlock_imagenet: QuantResBottleneckBlock_imagenet,
    BasicBlock_cifar10: QuantBasicBlock_cifar10,
    Bottleneck_cifar10: QuantBottleneck_cifar10,
    Block: QuantInvertedResidual_cifar10,
    ResBottleneckBlock_cifar10: QuantResBottleneckBlock_cifar10,
}

specials_unquantized = [nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d, nn.Dropout]