import torch
import torch.nn as nn
import torch.nn.functional as F

from .Ham_Layer.burger import get_hamburger


class HamConfig:
    """
    为 Hamburger 提供官方默认的超参数配置，
    这样就不需要去污染你主程序的 args 了！
    """
    HAM_TYPE = 'NMF'  # 使用最经典的非负矩阵分解
    MD_S = 1
    MD_D = 512  # 内部投影维度
    MD_R = 64  # 矩阵分解的 Rank
    TRAIN_STEPS = 6  # 训练时的迭代步数
    EVAL_STEPS = 7  # 测试时的迭代步数
    INV_T = 100
    ETA = 0.9
    RAND_INIT = True
    SPATIAL = True


class HamDecoderForMamba(nn.Module):
    """
    专为 MorphMamba 骨干网络设计的 Hamburger 解码器
    """

    def __init__(self, in_channels=256, num_layers=2, num_classes=1):
        super().__init__()
        self.num_layers = num_layers

        # 1. 特征融合层
        # 将传入的多层特征（比如 2 层，拼接后是 512）重新降维回 256
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_layers, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 2. Hamburger 核心注意力模块 (选用官方标准的 V2 版本)
        ham_class = get_hamburger('V2')
        self.hamburger = ham_class(in_c=in_channels, args=HamConfig())

        # 3. 预测头
        self.dropout = nn.Dropout2d(0.1)
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features):
        # features 是一个 list，包含了你 MorphEncoder 输出的多个 64x64 的特征图
        # 动态截取传入的特征
        valid_features = features[:self.num_layers]

        # 1. 沿通道维度拼接并降维融合 (尺寸依然是 64x64)
        x = torch.cat(valid_features, dim=1)
        x = self.fuse_conv(x)

        # 2. 通过 Hamburger 提取全局上下文注意力 (这步非常消耗算力，是它最大的劣势)
        x = self.hamburger(x)

        # 3. 预测类别
        x = self.dropout(x)
        logits = self.cls_head(x)

        # 4. 暴力放大回原图尺寸 512x512
        logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

        return logits