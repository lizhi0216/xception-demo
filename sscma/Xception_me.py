from typing import List, Optional, Tuple, Union

import torch.nn as nn
from mmengine.model import BaseModule
from torchvision.models._utils import _make_divisible

import sys
sys.path.append('/home/disk1/fxq/lianghua/ModelAssistant-main')

from sscma.models.base.general_add import ConvNormActivation, SeparableConv2d, Block

from sscma.registry import BACKBONES, MODELS
import math
import os
import torch
import torch.nn as nn
# import torch.utils.model_zoo as model_zoo


class Xception(BaseModule):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, downsample_factor=16):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super().__init__()

        stride_list = None
        if downsample_factor == 8:
            stride_list = [2,1,1]
        elif downsample_factor == 16:
            stride_list = [2,2,1]
        else:
            raise ValueError('xception.py: output stride=%d is not supported.'%os)
        norm_layer = nn.BatchNorm2d
        conv_layer = nn.Conv2d
        activation_layer = nn.ReLU

        # self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)#输入通道数，输出通道数，卷积核大小，步长=2--一次下采样，padding，偏差
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU(inplace=True)
        self.convbnrelu_1 = ConvNormActivation(in_channels=3, out_channels=32, kernel_size=3,
             stride=2, padding=1, bias=False, norm_layer=norm_layer, activation_layer=activation_layer)

        # self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
        # self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)
        self.convbnrelu_2 = ConvNormActivation(in_channels=32, out_channels=64, kernel_size=3,
                                               stride=1, padding=1, bias=False, norm_layer=norm_layer,
                                               activation_layer=None)
        #do relu here

        self.block1=Block(64,128,2)
        self.block2=Block(128,256,stride_list[0])
        self.block3=Block(256,728,stride_list[1])

        rate = 16//downsample_factor
        self.block4=Block(728,728,1,atrous=rate)
        self.block5=Block(728,728,1,atrous=rate)
        self.block6=Block(728,728,1,atrous=rate)
        self.block7=Block(728,728,1,atrous=rate)

        self.block8=Block(728,728,1,atrous=rate)
        self.block9=Block(728,728,1,atrous=rate)
        self.block10=Block(728,728,1,atrous=rate)
        self.block11=Block(728,728,1,atrous=rate)

        self.block12=Block(728,728,1,atrous=rate)
        self.block13=Block(728,728,1,atrous=rate)
        self.block14=Block(728,728,1,atrous=rate)
        self.block15=Block(728,728,1,atrous=rate)

        self.block16=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block17=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block18=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block19=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])

        self.block20=Block(728,1024,stride_list[2],atrous=rate,grow_first=False)
        self.conv3 = SeparableConv2d(in_channels=1024,out_channels=1536,kernel_size=3,stride=1,padding=1*rate,dilation=rate,activate_first=False)

        self.conv4 = SeparableConv2d(1536,1536,3,1,1*rate,dilation=rate,activate_first=False)

        self.conv5 = SeparableConv2d(1536,2048,3,1,1*rate,dilation=rate,activate_first=False)
        self.layers = []


    def init_weights(self):
        super().init_weights()
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input):#整体网络结构
        self.layers = []
        x = self.convbnrelu_1(input)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.convbnrelu_2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        # low_featrue_layer = self.block2.hook_layer#用于解码器融合深浅特征图
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)

        return x


def xception(pretrained=True, downsample_factor=16):#加载模型结构
    model = Xception(downsample_factor=downsample_factor)
    
    if pretrained:
        # 保存初始状态字典
        init_state_dict = model.state_dict()
        # 加载预训练权重
        pretrained_weights = torch.load('/home/disk1/fxq/lianghua/ModelAssistant-main/model_pre_weight/xception_pytorch_imagenet.pth')
        model.load_state_dict(pretrained_weights, strict=False)
        # model.load_state_dict(torch.load('/home/disk1/fxq/lianghua/ModelAssistant-main/model_pre_weight/xception_pytorch_imagenet.pth'), strict=False)
        # 计算加载了多少预训练权重
        changed_weights = 0
        total_weights = 0
        for key in init_state_dict.keys():
            total_weights += torch.numel(init_state_dict[key])
            if key in pretrained_weights and not torch.equal(init_state_dict[key], pretrained_weights[key]):
                changed_weights += torch.numel(init_state_dict[key])
        print(f'加载了 {100. * changed_weights / total_weights:.2f}% 的预训练权重')
    return model


if __name__ == "__main__":
    model = xception()
    x = torch.randn((1,3,512,512))
    out1, out2 = model(x)
    print(out1.shape)
