'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2



__all__ = ['LKA', 'hg']



class LKAModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttentionModule(nn.Module):
    def __init__(self, d_model=11):
        super().__init__()
        self.spatial_gating_unit = LKAModule(dim=d_model)

    def forward(self, x):
        shorcut = x.clone()
        x = self.spatial_gating_unit(x)
        x = x + shorcut
        return x
    

class MultiScaleLKAModule(nn.Module):
    def __init__(self, in_channels=256):
        """
        Initialize the Inception LKA (I-LKA) module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        """
        super().__init__()
        
        kernels = [3, 5, 7]
        paddings = [1, 4, 9]
        dilations = [1, 2, 3]
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.spatial_convs = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=kernels[i], stride=1,
                                                    padding=paddings[i], groups=in_channels,
                                                    dilation=dilations[i]) for i in range(len(kernels))])
        self.conv1 = nn.Conv2d(3*in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()
        attn = self.conv0(x)
        spatial_attns = [conv(attn) for conv in self.spatial_convs]
        attn = torch.cat(spatial_attns, dim=1)
        attn = self.conv1(attn)
        return original_input * attn
    
    
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        """
        Initialize the Attention module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        
        :return: Output tensor after applying attention module
        """
        super().__init__()
        self.proj_1x1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating = MultiScaleLKAModule(in_channels)
        self.proj_1x1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()
        x = self.proj_1x1_1(x)
        x = self.activation(x)
        x = self.spatial_gating(x)
        x = self.proj_1x1_2(x)        
        return x

