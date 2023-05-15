# Copyright (c) Xinzi He
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from transformer_block import Mlp
import numpy as np
class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.psp = PSPModule(sizes=(1, 2, 4, 16), dimension=2)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1, 4)  # 3 B heads N C
        q, k, v = qkv[0], qkv[1], qkv[2]
        v_B, v_head, v_N, v_C = q.shape
        v_pooled = self.psp(v.reshape(v_B * v_head, v_N, v_C)).view(v_B, v_head, -1, v_C)
        k = self.psp(k.reshape(v_B * v_head, v_N, v_C)).view(v_B, v_head, -1, v_C)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v_pooled).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatialTransformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(in_dim)
        self.attn = Attention(
            in_dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp1 = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp1(self.norm2(x)))
        return x


class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 4, 12, 32), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, hw, c  = feats.size()
        feats = feats.transpose(1,2).view(n,c,int(np.sqrt(hw)),int(np.sqrt(hw)))
        priors = [stage(feats).view(n, c, -1).transpose(1,2) for stage in self.stages]
        center = torch.cat(priors, -2)
        return center





