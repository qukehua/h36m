import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import seaborn as sns
# import palettable#python颜色库
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class TrajectoryNet(nn.Module):
    def __init__(self):
        super(TrajectoryNet, self).__init__()
        self.TB_foward_0 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.75)

        )
        self.TB_foward_1 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.75)

        )
        self.TB_foward_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.75)

        )
        self.TB_foward_3 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.75)

        )
        self.TB_foward_4 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.75)

        )
        self.TB_foward_5 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.75)

        )
        self.TB_foward_6 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.75)

        )
        self.TB_residual_0 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.TB_residual_1 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.TB_residual_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.TB_residual_3 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.layer_0 = torch.nn.Sequential(
            nn.Conv2d(10, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.layer_1 = torch.nn.Sequential(
            nn.Conv2d(64, 10, 3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(negative_slope=0.2)
            # nn.init.kaiming_normal_(nn.Conv2d.weight, mode='fan_in', nonlinearity='leaky_relu')

        )
        self.layer_2 = torch.nn.Sequential(
            nn.Conv2d(10, 10, 1, stride=1, padding=0),
        )
        self.transform = Transformer()
        self.TimeTransformer = TimeTransformer()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        batch_size = x.size(0)
        print(x.shape)
        #p = x.data.cpu().numpy().copy()
        x = self.layer_0(x)
        print(x.shape)
        transform_size = x.size(0) * x.size(1)
        print('transform_size',transform_size)
        transform_time_size = x.size(0) * x.size(2)
        print('transform_time_size',transform_time_size)
        transform_init = x[0,0,:,:]
        print('transform_init',transform_init.shape)
        transform_time_init = x[0,:,0,:]
        print('transform_time_init',transform_time_init.shape)
        transform_input = transform_init[None,:,:]
        transform_time_input = transform_time_init[None,:,:]
       # transform_out = transform_input
       # transform_time_out = transform_time_input
        print('transform_input',transform_input.shape)
        print('transform_time_input',transform_time_input.shape)
        for b in range(transform_size):
            transform_curr = self.transform(transform_input)

            if(b==0):
                transform_out=transform_curr
            else:
                transform_out = torch.cat([transform_out,transform_curr],dim = 0)


        for b in range(transform_time_size):
            transform_time_curr = self.TimeTransformer(transform_time_input);
            if(b==0):
                transform_time_out=transform_time_curr
            else:
                transform_time_out = torch.cat([transform_time_out,transform_time_curr],dim = 0)
        print('transform_time_out',transform_time_out.shape)
        transform_time_out = transform_time_out.reshape(int(x.size(0)),int(x.size(2)),64,3)
        print('transform_time_out', transform_time_out.shape)
        transform_time_out = torch.transpose(transform_time_out,1,2)
        print('transform_time_out',transform_time_out.shape)
        print('transform_out',transform_out.shape)
        transform_out = transform_out.reshape(int(x.size(0)),int(x.size(1)),22,3)
        print('transform_out', transform_out.shape)
        res = transform_out+transform_time_out
        print('res',res.shape)


        for i in range(1):
            x = self.TB(res)
        x = self.layer_1(x)
        x = self.layer_2(x)
        print('x',x.shape)
        return x #p

    def TB(self, x):
        batch_size = x.size(0)
        x_0 = self.TB_residual_0(x)
        traj_1 = self.TB_foward_0(x)
        x_1 = self.TB_residual_1(traj_1)
        traj_2 = self.TB_foward_1(traj_1)
        x_2 = self.TB_residual_2(traj_2)
        traj_3 = self.TB_foward_2(traj_2)
        x_3 = self.TB_residual_3(traj_3)
        traj_4 = self.TB_foward_3(traj_3)
        traj_5 = self.TB_foward_4(traj_4 + x_3)
        traj_6 = self.TB_foward_5(traj_5 + x_2)
        traj_7 = self.TB_foward_6(traj_6 + x_1)
        out = traj_7 + x_0
        # transform_size = out.size(0) * out.size(1)
        # transform_init = x[0,0,:,:]
        # transform_input = transform_init[None,:,:]
        # transform_out = transform_input
        # for b in range(transform_size - 1):
        #     transform_curr = self.transform(transform_input)
        #     transform_out = torch.cat([transform_out,transform_curr],dim = 0)
        # res1 = transform_out.reshape(int(out.size(0)),int(out.size(1)),22,3)
        return out
class Transformer(nn.Module):
    def __init__(self, num_joints=22, in_chans=3, embed_dim_ratio=3, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, attn_drop_rate=0.75,drop_rate= 0.1):
        super(Transformer, self).__init__()
        norm_layer =  partial(nn.LayerNorm, eps=1e-6)
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.pos_drop = nn.Dropout(p=drop_rate)
    def forward(self, x):
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        return x
class TimeTransformer(nn.Module):
    def __init__(self, num_joints=64, in_chans=3, embed_dim_ratio=3, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, attn_drop_rate=0.75,drop_rate= 0.1):
        super(TimeTransformer, self).__init__()
        norm_layer =  partial(nn.LayerNorm, eps=1e-6)
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.pos_drop = nn.Dropout(p=drop_rate)
    def forward(self, x):
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim = 8, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.75, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        #self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        b, j, c = x.shape
        qkv = self.qkv(x).reshape(b, j, 3, 1, c ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_heap = attn[0,0,:,:]
        # print('attention',attn_heap.shape)
        x = (attn @ v).transpose(0, 1).reshape(b, j, c)
        x_heap = x[0,:,0]
        x_heap = x_heap.unsqueeze(1)
        sum_heap = attn_heap + x_heap
        # plt.figure(dpi=120)
        # sns.heatmap(data=sum_heap.cpu().detach().numpy(),fmt="d",cmap='YlGnBu')
        # plt.title('所有参数默认')
        # plt.show()
        # plt.savefig('squares_plot.png', bbox_inches='tight')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, attn_drop=0.75
                 , act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        self.drop_path =  nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        heap2 = x[0,:,0]
        return x