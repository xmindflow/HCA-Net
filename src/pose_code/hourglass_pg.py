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
from .embedding import SinusoidalPositionEmbeddings
from torchvision.transforms import GaussianBlur


__all__ = ['HourglassNet', 'hg']



normalize = lambda x: (x-x.min())/(x.max()-x.min())

class LKA_module(nn.Module):
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


class SpatialAttention(nn.Module):
    def __init__(self, d_model=11):
        super().__init__()
        self.spatial_gating_unit = LKA_module(dim=d_model)

    def forward(self, x):
        shorcut = x.clone()
        x = self.spatial_gating_unit(x)
        x = x + shorcut
        return x
    

class multisclae_LKAModule(nn.Module):
    def __init__(self, in_channels=256):
        """
        Initialize the Inception LKA (I-LKA) module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        """
        super(multisclae_LKAModule, self).__init__()
        
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
    def __init__(self, in_channels=256):
        """
        Initialize the Attention module.

        :param in_channels: Number of input channels.
        :type in_channels: int
        
        :return: Output tensor after applying attention module
        """
        super(AttentionModule, self).__init__()
        self.proj_1x1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating = multisclae_LKAModule(in_channels)
        self.proj_1x1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()
        x = self.proj_1x1_1(x)
        x = self.activation(x)
        x = self.spatial_gating(x)
        x = self.proj_1x1_2(x)        
        return x


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class SGG(nn.Module):
    """Skeleton-based Guidance Generator
    Parameters:
        - x: block
    Output:
        - sfv: skeleton feature vector
    """
    def __init__(self, in_channels, hidden=64, sampling_times=10):
        super().__init__()
        self.sampling_times = sampling_times
        self.smooth = GaussianBlur(5, sigma=(0.5, 0.5))
        self.smooth_large = GaussianBlur(11, sigma=(5, 5))
        self.fc1 = nn.Linear(in_channels, hidden, bias=True)
        self.act = nn.PReLU()
        self.fc2 = nn.Linear(hidden, in_channels, bias=False)
        self.gelu = nn.GELU()

    def _choose_joints(self, x):

        b, c, h, w = x.shape
        
        x = normalize(x).clone()
        x = torch.where(x<0.3, 0, x)
        
        if np.random.rand() > 0.9:
            att = x[0].sum(0).detach().cpu()
            att = normalize(att)
            cv2.imwrite("att.png", np.uint8(att*255))
        
        urnd = torch.rand_like(x)
        crnd = x * urnd
        
        crnd_reshaped = crnd.view(b, c, -1) 
        argmax_indices = torch.argmax(crnd_reshaped, dim=2)

        i_indices = argmax_indices // w
        j_indices = argmax_indices %  w

        i_indices = i_indices.view(b, c, 1) / h
        j_indices = j_indices.view(b, c, 1) / w

        y = torch.cat((i_indices, j_indices), dim=2)
        vis = torch.zeros([b, c])==0
        for ib in range(b):
            for ic in range(c):
                row = torch.round(y[ib,ic,0]*h).type(torch.int)
                col = torch.round(y[ib,ic,1]*w).type(torch.int)
                if crnd[ib,ic,row,col] < 0.001:
                    y[ib,ic,:] = 0*y[ib,ic,:]
                    vis[ib,ic]=False

        vis = y
        _extra = torch.zeros([b, c + c*(c-1)//2, 2])
        for ib in range(b):
            for ic in range(c-1):
                for jc in range(ic+1,c):
                    if not (vis[ib,ic] or vis[ib,jc]):
                        a = y[ib,ic,:]
                        b = y[ib,jc,:]
                        dist = (a - b).pow(2).sum().sqrt()
                        rad = torch.angle((a[1]-b[1])+1j*(a[0]-b[0]))/torch.pi
                        _extra[ib, c + ]=[dist,rad]
        _extra = nu            
        
        noise = torch.randn_like(y)
        
        # y = y.reshape(b, -1)
        # y = self.act(self.fc1(y))
        # y = self.fc2(y)
        # y = y.reshape(b, c, 2)
        joints = self.gelu(y)
        
        joints =  y
        
        res = joints + 0.01*noise
        return res.reshape(b, c , 2)

    def _fill_points(self, att, joints, value=1):
        b, c, h, w = att.shape
        # tmp =  []
        for bb in range(b):
            for cc in range(c):
                i,j = joints[bb,cc]
                row = torch.clip(torch.round(j*h).type(torch.int), 0, h-1) 
                col = torch.clip(torch.round(i*w).type(torch.int), 0, w-1)
                # tmp.append((row.item(), col.item()))
                att[bb, cc, row, col] = value
        # print(tmp)
        # b_indices = torch.arange(b)
        # c_indices = torch.arange(c)
        # att[b_indices, c_indices, joints[:, :, 0], joints[:, :, 1]] = value
        return att

    def forward(self, x):
        x = self.smooth(x)

        rjoints = torch.stack([self._choose_joints(x) for _ in range(self.sampling_times)], 0)
        v_joints, m_joints = torch.var_mean(rjoints, 0)

        att = torch.zeros_like(x)+0.001
        att = self._fill_points(att, m_joints)
        att = self.smooth_large(att)
        # att = normalize(att)
        return att, m_joints, v_joints


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        
        # build hourglass modules
        ch = self.num_feats*block.expansion
        
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.scale_score = nn.Upsample(scale_factor=4, mode='bilinear')
        self.down4xconv = nn.Conv2d(num_classes, ch, 7, 4, 3)

        hg, res, fc, score, out, sgg, fc_, score_, att_layer = [], [], [], [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            att_layer.append(AttentionModule())
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            out.append(nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear'),
                SpatialAttention(d_model=num_classes)
            ))
            sgg.append(SGG(num_classes, hidden=64, sampling_times=10))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.sgg = nn.ModuleList(sgg)
        self.out = nn.ModuleList(out)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
        self.att_layer = nn.ModuleList(att_layer)
        # self.spatial_refinement = SpatialAttention(d_model=num_classes)
        
        
    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        outs, sgis = [], []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.att_layer[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)

            scaled_score = self.out[i](score)
            outs.append(scaled_score)

            satt, mean_js, var_js = self.sgg[i](scaled_score)
            sgis.append([satt, mean_js, var_js])

            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + (fc_ + score_) #* self.down4xconv(satt)

        return outs, sgis


def hg(**kwargs):
    model = HourglassNet(block=Bottleneck, 
                         num_stacks=kwargs['num_stacks'], 
                         num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
