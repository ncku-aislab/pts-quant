from .imagenet.ResNet import resnet18_imagenet, resnet50_imagenet, BasicBlock_imagenet, Bottleneck_imagenet
from .imagenet.MobileNetV2 import mobilenetv2_imagenet, InvertedResidual_imagenet
from .imagenet.regnet import regnetx_600m, regnetx_3200m

from .cifar10.ResNet import resnet18_cifar10, resnet50_cifar10, BasicBlock_cifar10, Bottleneck_cifar10
from .cifar10.MobileNetV2 import mobilenetv2_cifar10, Block
from .cifar10.regnet import regnetx_200m, regnetx_400m