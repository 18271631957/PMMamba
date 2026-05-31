import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """标准的 UNet 双层卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 加上 BN 让训练更稳定
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetDecoderForMamba(nn.Module):
    """专为各向同性 (Isotropic) Mamba 骨干设计的 UNet 解码器"""

    def __init__(self, MASSLayer_out_dim, num_classes=1):
        super().__init__()
        # MASSLayer_out_dim 是你 MorphEncoder 输出的通道数 256

        # Stage 1: 从 64x64 上采样到 128x128
        self.up1 = nn.ConvTranspose2d(MASSLayer_out_dim, 256, kernel_size=2, stride=2)
        # 拼接跳跃连接后，通道数为 256 + MASSLayer_out_dim
        self.conv1 = DoubleConv(256 + MASSLayer_out_dim, 256)

        # Stage 2: 从 128x128 上采样到 256x256
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128 + MASSLayer_out_dim, 128)

        # Stage 3: 从 256x256 上采样到 512x512
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64 + MASSLayer_out_dim, 64)

        # 最终预测头
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features):
        # 动态获取特征层数 K
        K = len(features)

        # 1. 确定解码起点（始终使用最深的一层）
        x = features[-1]

        # 2. Stage 1: 从 64x64 上采样到 128x128
        # 逻辑：尝试取倒数第二层，如果没有则复用当前层
        skip_idx_1 = max(0, K - 2)
        x = self.up1(x)
        skip1 = F.interpolate(features[skip_idx_1], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv1(x)

        # 3. Stage 2: 从 128x128 上采样到 256x256
        # 逻辑：尝试取更浅的层，最差情况复用最浅层
        skip_idx_2 = max(0, K - 3)
        x = self.up2(x)
        skip2 = F.interpolate(features[skip_idx_2], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)

        # 4. Stage 3: 从 256x256 上采样到 512x512
        # 逻辑：固定取最浅层（索引0），因为它对边缘恢复最关键
        x = self.up3(x)
        skip3 = F.interpolate(features[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv3(x)

        return self.outc(x)

    # def forward(self, features):
    #     # features 是你 backbone 提取出的多层特征集合，假设有4个，尺寸全是 (B, C, 64, 64)
    #     f1, f2, f3, f4 = features[0], features[1], features[2], features[-1]
    #
    #     # 将最深层特征 f4 作为 UNet 解码的起点
    #     x = f4
    #
    #     # 第一层上采样与拼接 (64x64 -> 128x128)
    #     x = self.up1(x)
    #     skip3 = F.interpolate(f3, size=x.shape[2:], mode='bilinear', align_corners=False)
    #     x = torch.cat([x, skip3], dim=1)  # 跳跃连接拼接
    #     x = self.conv1(x)
    #
    #     # 第二层上采样与拼接 (128x128 -> 256x256)
    #     x = self.up2(x)
    #     skip2 = F.interpolate(f2, size=x.shape[2:], mode='bilinear', align_corners=False)
    #     x = torch.cat([x, skip2], dim=1)
    #     x = self.conv2(x)
    #
    #     # 第三层上采样与拼接 (256x256 -> 512x512)
    #     x = self.up3(x)
    #     skip1 = F.interpolate(f1, size=x.shape[2:], mode='bilinear', align_corners=False)
    #     x = torch.cat([x, skip1], dim=1)
    #     x = self.conv3(x)
    #
    #     # 输出 512x512 的掩膜
    #     logits = self.outc(x)
    #     return logits