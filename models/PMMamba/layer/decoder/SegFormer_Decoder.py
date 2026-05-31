import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear Embedding (原汁原味从 SegFormer 官方提取)
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # 展平空间维度，进行线性映射
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerDecoderForMamba(nn.Module):
    """
    专为各向同性 (Isotropic) Mamba 骨干设计的 SegFormer (All-MLP) 解码器
    """

    def __init__(self, in_channels, decoder_dim=256, num_classes=1, num_layers=4):
        super().__init__()
        self.num_layers = num_layers

        # 1. 为每一层特征创建一个 MLP 投影层
        # 因为 MorphMamba 所有层的通道数是不变的 (等于 in_channels)，所以 input_dim=in_channels
        self.mlps = nn.ModuleList([
            MLP(input_dim=in_channels, embed_dim=decoder_dim) for _ in range(num_layers)
        ])

        # 2. SegFormer 的特征融合层
        # 拼接后通道数为 decoder_dim * num_layers
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * num_layers, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(0.1)

        # 3. 预测头
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features):
        # features 是你 backbone 提取出的多层特征集合
        # 假设传入了 num_layers 层，且尺寸全是 (B, in_channels, 64, 64)

        outs = []
        for i in range(self.num_layers):
            feat = features[i]
            n, _, h, w = feat.shape

            # 将特征展平、降维并重塑回 (B, decoder_dim, 64, 64)
            _c = self.mlps[i](feat).permute(0, 2, 1).reshape(n, -1, h, w)
            outs.append(_c)

        # ====== 核心优势 ======
        # 因为 MorphMamba 所有特征都是 64x64，所以我们不需要像官方代码那样 resize！
        # 直接沿通道维度暴力拼接
        _c = self.linear_fuse(torch.cat(outs, dim=1))

        # Dropout + 预测
        x = self.dropout(_c)
        logits = self.linear_pred(x)

        # ====== 恢复原图分辨率 ======
        # 你的 Ground Truth 应该是 512x512，所以我们要把 64x64 的 logits 双线性插值放大
        logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

        return logits