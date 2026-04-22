
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      minifasnet.py
# .:ccccccccccccc;KMMc;cc;xMMc;ccccccc:.    Description  :      MiniFASNet architecture for liveness detection.
# ,cccccccccccccc;MMM.;cc;;WW:;cccccccc,    --------------
# :cccccccccccccc;MMM.;cccccccccccccccc:
# :ccccccc;oxOOOo;MMM000k.;cccccccccccc:
# cccccc;0MMKxdd:;MMMkddc.;cccccccccccc;
# ccccc;XMO';cccc;MMM.;cccccccccccccccc'
# ccccc;MMo;ccccc;MMW.;ccccccccccccccc;
# ccccc;0MNc.ccc.xMMd;ccccccccccccccc;
# cccccc;dNMWXXXWM0:;cccccccccccccc:,
# cccccccc;.:odl:.;cccccccccccccc:,.
# ccccccccccccccccccccccccccccc:'.
# :ccccccccccccccccccccccc:;,..
#  ':cccccccccccccccc::;,.


# import necessary libraries
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Standard Convolution Block:
        Conv2D → BatchNorm → ReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size
        stride: Stride
        padding: Padding
        groups: Group convolution (used for depthwise conv)

    Purpose:
        Basic building block for feature extraction.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class DepthWise(nn.Module):
    """
    Depthwise Separable Convolution Block:
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the depthwise convolution
    Purpose:
        Reduce computational cost compared to standard convolution while maintaining representation power.
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.block(x)
    
class Residual(nn.Module):
    """
    Residual Block:
        Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU

    Args:
        c: Number of channels
        number_blocks: Number of convolutional blocks in the residual unit

    Purpose:
        Introduce skip connections to facilitate training of deep networks.
    """

    def __init__(self, c, number_blocks):
        super().__init__()
        layers = []

        for _ in range(number_blocks):
            layers.append(ConvBlock(c, c, stride=1))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)
    
class MiniFASNet(nn.Module):
    """
    MiniFASNet for Liveness Detection:
    Args:
        num_classes: Number of output classes (default is 2 for binary classification)
    Purpose:
        A compact and efficient architecture designed for liveness detection tasks, utilizing depthwise separable convolution.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        
        self.conv1 = ConvBlock(3, 64, 3, 2)
        self.conv2_dw = DepthWise(64, 64, 1)

        self.conv_23 = DepthWise(64, 128, 2)
        self.conv_3 = Residual(128, 4)

        self.conv_34 = DepthWise(128, 128, 2)
        self.conv_4 = Residual(128, 6)

        self.conv_45 = DepthWise(128, 128, 2)
        self.conv_5 = Residual(128, 2)

        self.conv_6_sep = ConvBlock(128, 512, kernel_size=1, stride=1, padding=0)
        self.conv_6_dw = DepthWise(512, 512, kernel_size=7, stride=1, padding=0, groups=512)

        self.conv_6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)

        x = self.conv_23(x)
        x = self.conv_3(x)

        x = self.conv_34(x)
        x = self.conv_4(x)

        x = self.conv_45(x)
        x = self.conv_5(x)

        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)

        x = self.conv_6_flatten(x)
        x = self.linear(x)

        return x
