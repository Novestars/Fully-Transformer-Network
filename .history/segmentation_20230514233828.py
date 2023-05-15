from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from SpatialTransformer import SpatialTransformer
from transformer_block import Block
from torch import nn
import torch
sigmoid = nn.Sigmoid()

class FTN_encoder(nn.Module):
    """
    FTN encoding module
    """
    def __init__(self, img_size=512,  in_chans=3, token_dim=64):
        super().__init__()
        self.token_dim = token_dim
        self.swt_0 = nn.Conv2d(in_chans,token_dim,kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        self.swt_1 = nn.Conv2d(token_dim,token_dim*2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.swt_2 = nn.Conv2d(token_dim*2,token_dim*4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.attention1 = SpatialTransformer(dim=token_dim, in_dim=token_dim, num_heads=token_dim//32, mlp_ratio=1.0,attn_drop = 0.0,drop_path=0,drop=0.)
        self.attention2 = SpatialTransformer(dim=token_dim*2, in_dim=token_dim*2, num_heads=token_dim//16, mlp_ratio=1.0,attn_drop = 0.0,drop_path=0,drop=0.)

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        encoder_list = []
        B = x.shape[0]
        x = self.swt_0(x).view(B,self.token_dim,-1).transpose(1, 2)
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        encoder_list.append(x)
        x = self.swt_1(x).view(B,self.token_dim*2,-1).transpose(1, 2)
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        encoder_list.append(x)
        x = self.swt_2(x).view(B,self.token_dim*4,-1).transpose(1, 2)

        return x,encoder_list

class FTN_decoder(nn.Module):
    """
    FTN decoding module
    """
    def __init__(self, img_size=512, token_dim=64):
        super().__init__()
        self.token_dim = token_dim

        self.proj = nn.Sequential(nn.Linear(token_dim*4,self.token_dim*2))

        self.attention1 = SpatialTransformer(dim=token_dim*4, in_dim=token_dim*4, num_heads=1, mlp_ratio=1.0,attn_drop = 0.0,drop_path=0,drop=0.)
        self.attention2 = SpatialTransformer(dim=token_dim*2, in_dim=token_dim*2, num_heads=1, mlp_ratio=1.0,attn_drop = 0.0,drop_path=0,drop=0.)
        self.swt_0 = nn.Sequential(nn.Conv2d(token_dim*2,token_dim*4, kernel_size=(3, 3),  padding=(1, 1)))

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x, encoder_list):
        B = x.shape[0]
        x= self.proj(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        x = torch.cat([encoder_list[-1],self.upsample(x)],1).view(B,self.token_dim*2,-1).transpose(1, 2)
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = self.swt_0(x)
        x = torch.cat([encoder_list[-2],self.upsample(x)],1).view(B,self.token_dim*2,-1).transpose(1, 2)
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        return x


class FTN(nn.Module):
    def __init__(self, img_size=448,  in_chans=3, num_classes=9, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0,
                 drop_path_rate=0, norm_layer=nn.LayerNorm, token_dim=64,use_meta = False):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = FTN_encoder(
                img_size=img_size,  token_dim=token_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=token_dim*4, in_dim=token_dim*4,num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.decoder = FTN_decoder(
                img_size=img_size,  token_dim=token_dim)

        self.pre_hade_norm = nn.InstanceNorm2d(token_dim*2)
        self.classifier = nn.Conv2d(token_dim*2, num_classes,kernel_size=3,padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Dim_reduce
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x,encoder_lsit = self.encoder(x)
        for blk in self.blocks:
            x = blk(x)
        return x, encoder_lsit

    def forward(self, x):
        x,encoder_lsit = self.forward_features(x)
        x = self.decoder(x,encoder_lsit)
        x = self.pre_hade_norm(x)
        out = self.classifier(x)
        return out







if __name__ == '__main__':
    x = torch.randn(2, 3, 128,128)
    model = FTN()
    y = model(x)
    print(y.shape)