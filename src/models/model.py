import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from functools import partial
from torchvision.transforms import GaussianBlur
from modules.attentions import AttentionModule, SpatialAttentionModule
from modules.ips import IPS
from modules.hg import Bottleneck, Hourglass



__all__ = ['HCANet', ]



class HCANet(nn.Module):
    def __init__(self, block=Bottleneck, num_stacks=2, num_blocks=4, num_classes=11,
                 ips=True, ips_feedback=True, ips_feedback_sum=False):
        super(HCANet, self).__init__()
        
        ips_block = partial(IPS, sampling_times=5)
        self.ips = ips
        self.ips_feedback = ips_feedback
        self.ips_feedback_sum = ips_feedback_sum

        self.inplanes = 32
        self.num_feats = 64
        
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

        hg, res, fc, score, out, ips, fc_, score_, att_layer = [], [], [], [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            att_layer.append(AttentionModule(in_channels=128))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            out.append(nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear'),
                SpatialAttentionModule(d_model=num_classes)
            ))
            ips.append(ips_block(num_classes))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.ips = nn.ModuleList(ips)
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

            satt, mean_js, var_js, vis_js = self.ips[i](scaled_score)
            sgis.append([satt, mean_js, var_js, vis_js])

            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + self.finalize_hca(score_, satt)

        return outs, sgis
    
    
    def finalize_hca(self, score, satt):
        if self.ips and self.ips_feedback:
            if self.ips_feedback_sum:
                return score+self.down4xconv(satt)
            else:
                return score*self.down4xconv(satt)
        else:
            return score
