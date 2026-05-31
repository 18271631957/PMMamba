import copy
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# 【重要提醒】这里的导入路径请根据你本地纯净版的文件夹结构进行微调
from .embed import resize_pos_embed
from .patch_embed import ConvPatchEmbed
from .MASS_Layer import MASS_Layer


class MorphEncoder(nn.Module):
    def __init__(self, args, img_size=512, in_channels=3,
                 embed_dims=256,  # 每一块拼图的特征向量长度
                 # embed_dims=128,  # 每一块拼图的特征向量长度
                 num_layers=4,  # Mamba 车间的层数
                 patch_size=8,  # 切块大小 8x8
                 num_convs_patch_embed=2,  # 切图机预处理卷积层数
                 mamba_d_state=16,  # Mamba 内部的状态维度
                 mamba_expand=2,  # Mamba 内部的扩展因子
                 out_indices=(0, 1, 2, 3),
                 drop_rate=0.,
                 drop_path_rate=0.2):

        super().__init__()

        self.num_layers = num_layers
        self.out_indices = out_indices

        # 实例化切图机
        self.patch_embed = ConvPatchEmbed(in_channels=in_channels, input_size=img_size, embed_dims=embed_dims, num_convs=num_convs_patch_embed, patch_size=patch_size, stride=patch_size)
        # 初始patch分辨率 (64,64)
        self.patch_resolution = self.patch_embed.init_out_size
        # 64*64=4096个patch
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # 制造 GPS 定位器并当场初始化
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Mamba 车间组装
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = nn.ModuleList()

        # 【极其优雅的图纸内联化】直接在这里定义基础图纸
        base_layer_cfg = {
            'mamba_cfg': {
                'mass_conv_type': args.mass_conv_type,
                'd_state': mamba_d_state,
                'expand': mamba_expand,
                'conv_size': 7,
                'bias': True,
            }
        }
        # DSASS
        for i in range(self.num_layers):
            _layer_cfg_i = copy.deepcopy(base_layer_cfg)
            _layer_cfg_i.update({
                "embed_dims": embed_dims,
                "layer_conv_times": args.layer_conv_times[i],
                'use_res': args.use_res,
                'use_synergy': args.use_synergy,
                'use_layer_conv': args.use_layer_conv,
                "drop_path_rate": dpr[i],
            })
            self.layers.append(MASS_Layer(**_layer_cfg_i))

    def forward(self, x):  # 假设输入 (1,3,512,512)
        # 切图机器  (1,3,512,512) -> (1,4096,256) (64,64)
        x, patch_resolution = self.patch_embed(x)
        pos_embed = resize_pos_embed(self.pos_embed, self.patch_resolution, patch_resolution, mode='bicubic', num_extra_tokens=0)
        x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            # (1, 4096, 256) -> (1, 4096, 256)
            x = layer(x, hw_shape=patch_resolution)

            if i in self.out_indices:
                B, _, C = x.shape  # 4, 4096, 256
                # 把 1D 的长条序列重新折叠回 2D 的空间网格 (4, 4096, 256) -> (4, 64, 64, 256)
                patch_token = x.reshape(B, *patch_resolution, C)
                # (1, 64, 64, 256) -> (1, 256, 64, 64)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs.append(patch_token)

        return outs
