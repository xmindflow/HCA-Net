import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from torchvision.transforms import GaussianBlur



__all__ = ['IPS']



normalize = lambda x: (x-x.min())/(x.max()-x.min())


class IPS(nn.Module):
    def __init__(self, in_channels, sampling_times=5):
        super().__init__()
        self.sampling_times = sampling_times
        # self.smooth = GaussianBlur(5, sigma=(0.1, 0.1))
        self.smooth_large = GaussianBlur(7, sigma=(1.5, 1.5))

    def _sample_joints(self, x):
        b, c, h, w = x.shape
        x = x.clone()
        # x = torch.where(x<0.5, 0, x)
        
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
                if crnd[ib,ic,row,col] < 0.1:
                    y[ib,ic,:] = 0*y[ib,ic,:]
                    vis[ib,ic]=False
        joints = y
        return joints, vis

    def _fill_points(self, att, joints, vis, value=1):
        b, c, h, w = att.shape
        for bb in range(b):
            for cc in range(c):
                if vis[bb,cc]:
                    i,j = joints[bb,cc]
                    row = torch.clip(torch.round(i*h).type(torch.int), 0, h-1) 
                    col = torch.clip(torch.round(j*w).type(torch.int), 0, w-1)
                    att[bb, cc, row, col] = value
        return att

    def forward(self, _x_):
        x = F.sigmoid(_x_).clone()
        
        # sample joints as a skeleton for a given sampling_times
        vis_list, joints_list = [], []
        for _ in range(self.sampling_times):
            joints, vis = self._sample_joints(x)
            joints_list.append(joints)
            vis_list.append(vis)
        
        v_joints, m_joints = torch.var_mean(torch.stack(joints_list, 0), 0)
        vis = torch.mean(torch.stack(vis_list, 0).float(), 0) > 0.5

        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                if vis[b, c]==False:
                    m_joints[b, c, :] *= 0
                    v_joints[b, c] *= 0
        
        att = torch.zeros_like(x)
        att = self._fill_points(att, m_joints, vis)
        att = self.smooth_large(att)
        att = normalize(att)
        
        return att, m_joints, v_joints, vis
