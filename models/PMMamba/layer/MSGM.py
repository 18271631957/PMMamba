import torch
import torch.nn as nn

class MorphMixer(nn.Module):
    """
    原创模块：形态学空间门控混合器 (Morphological Spatial Gating Mixer)
    替代臃肿且缺乏空间感知的双循环 GBC 模块。
    """
    def __init__(self, dim, expansion_factor=2):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        # 1. 升维投影 (Channel Mixing)
        self.proj_up = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=16, num_channels=hidden_dim)
        self.act = nn.GELU()

        # 2. 大核空间混合 (Large-Kernel Spatial Mixing)
        # 用 7x7感受野更大，对细小裂缝的包络性更好
        self.spatial_mixer = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=7, 
            padding=3, groups=hidden_dim, bias=False
        )

        # 3. [核心创新] 十字形态学空间门控 (Cross-Morpho Spatial Gate)
        # 生成二维空间上的注意力 Mask，精准高亮裂缝所在的像素
        self.gate_h = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 7), padding=(0, 3), groups=hidden_dim)
        self.gate_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(7, 1), padding=(3, 0), groups=hidden_dim)
        
        # 4. 降维投影 (Channel Mixing)
        self.norm2 = nn.GroupNorm(num_groups=16, num_channels=hidden_dim)
        self.proj_down = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        # 记录残差
        identity = x

        # 升维并激活
        x_up = self.act(self.norm1(self.proj_up(x)))

        # 提取局部 2D 大感受野特征
        x_spatial = self.spatial_mixer(x_up)

        # 生成形态学空间门控 Mask (十字相加后求 Sigmoid)
        gate = torch.sigmoid(self.gate_h(x_up) + self.gate_v(x_up))
        # 空间门控相乘：将非裂缝区域的特征直接压制为 0
        x_gated = x_spatial * gate

        # ====== 【核心：开后门存下提纯前、门控、提纯后的特征】 ======
        self.debug_spatial_feat = x_spatial.detach()  # 提纯前 (背景有很多噪声)
        self.debug_gate = gate.detach()  # 提纯工具 (只亮裂缝的掩膜)
        self.debug_gated_feat = x_gated.detach()  # 提纯后 (噪声被压制)
        # =======================================================

        # 降维输出并融合残差
        out = self.proj_down(self.norm2(x_gated))
        
        return out + identity