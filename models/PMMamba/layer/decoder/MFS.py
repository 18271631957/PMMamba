import torch.nn as nn
import torch

from ..DySample import DySample


class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        # 1. 顶层面包：高压浓缩机 (Pointwise) 作用：降维
        # 这就好比在同一个时间步上，用一个全连接层把你输入的几十个气象变量（风、光、温等），通过加权求和，浓缩成几个最核心的隐变量。它根本不看过去和未来（也就是不看周围像素），只做跨通道的信息融合。
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        # 2. 中间夹心：独立质检员 (Depthwise) 作用：在被压缩的特征图上，提取空间纹理（看周围的像素）。
        """
            当普通的 Conv2d 设置了分组数（groups）等于输入通道数时，它就变成了“深度卷积”。
            这意味着，原本一个卷积核要同时看红、绿、蓝所有通道，现在变成了每个通道配备一个专属的卷积核，各看各的，互不干扰。这让参数量和计算量瞬间断崖式下跌！
        """
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        # 3. 底层面包：解压融合机 (Pointwise) 作用：升维与最终融合。
        # 把中间夹心层各自独立提取出来的特征再次打乱、混合，并把厚度重新恢复或映射到目标厚度 out_channels
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        # 初始 x 形状: (B, 32, H, W)  <- 假设 in_channels = 32
        # 1. 穿过顶层 1x1 卷积：通道数被无情压缩 (32 -> 4)
        # (B, 32, H, W) -> (B, 4, H, W)
        x = self.pointwise_1(x)
        # 2. 穿过中间 KxK 深度卷积：各通道独立提取空间特征，通道数和分辨率通常不变
        # (B, 4, H, W) -> (B, 4, H, W)
        x = self.depthwise(x)
        # 3. 穿过底层 1x1 卷积：通道数被重新扩展 (4 -> 32)
        # (B, 4, H, W) -> (B, 32, H, W)  <- 假设 out_channels = 32
        x = self.pointwise_2(x)
        return x


class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()
        # 第一套工具：空间特征扫描仪 (视野 3x3)
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            nn.GroupNorm(num_groups=in_channels // 16, num_channels=in_channels),
            nn.ReLU()
        )
        # 第二套工具：空间特征深度扫描仪 (串联放大视野)
        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            nn.GroupNorm(num_groups=in_channels // 16, num_channels=in_channels),
            nn.ReLU()
        )
        # 第三套工具：通道权重评估员 (视野 1x1，只看深度不看周围)
        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            nn.GroupNorm(num_groups=in_channels // 16, num_channels=in_channels),
            nn.ReLU()
        )
        # 第四套工具：终极特征调和器 (视野 1x1，收尾打磨)
        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            nn.GroupNorm(num_groups=16, num_channels=in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # 初始 x 形状: (B, 32, 512, 512)
        residual = x
        # ============ 路线一：提取空间纹理 (Spatial Features) ============
        # 经过连续两次 3x3 瓶颈卷积，深刻理解裂缝在平面上的走向和边缘
        # 形状演变: (B, 32, 512, 512) -> (B, 32, 512, 512)
        x1 = self.block1(x)
        x1 = self.block2(x1)

        # ============ 路线二：评估通道权重 (Channel Attention/Gating) ============
        # 用 1x1 卷积纵向打分，评估这 32 层特征里，哪些层是噪音，哪些层是真正的裂缝
        x2 = self.block3(x)

        # ============ 核心魔法：逐元素门控相乘 ============
        # x2 就像一个遮罩 (Mask)，数值接近 0 的地方代表噪音，接近 1 的地方代表高价值特征。
        # 两者相乘，瞬间把 x1 提取出来的空间特征中的背景噪音全部抹除，只留下最纯粹的裂缝！
        x = x1 * x2

        # ============ 路线汇合：终极融合与调和 ============
        # 乘法之后的特征比较“生硬”，用 1x1 卷积再次打乱融合，让通道间的信息充分交流
        x = self.block4(x)

        # ============ 收尾：残差连接 (Residual Connection) ============
        # 把经过千锤百炼的 x，和一开始备份的 residual 加在一起。
        # 保证模型即使在极端情况下（比如相乘时把特征不小心乘没了），也能兜底保留原始信息，防止梯度消失。
        return x + residual


class MFS(nn.Module):
    def __init__(self, use_dysample=False, MASSLayer_out_dim=256, embedding_dim=8, num_layers=2):
        super(MFS, self).__init__()
        # 通道统一降维
        self.embedding_dim = embedding_dim  # 值是 8
        self.num_layers = num_layers

        # 1. 统一降维 (256 -> 8)
        self.linears = nn.ModuleList([
            nn.Linear(MASSLayer_out_dim, embedding_dim) for _ in range(self.num_layers)
        ])

        # 2. 统一放大 8 倍 (64x64 -> 512x512)
        self.dysamples = nn.ModuleList([
            DySample(embedding_dim, scale=8, dyscope=use_dysample) for _ in range(self.num_layers)
        ])

        # 3. MFS 核心融合模块
        self.GBC_C = GBC(embedding_dim * self.num_layers)
        self.linear_fuse = BottConv(embedding_dim * self.num_layers, embedding_dim, embedding_dim // 8, kernel_size=1, padding=0, stride=1)

        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        outs = []
        for i in range(self.num_layers):
            c = inputs[i]
            b, ch, h, w = c.shape
            # 降维投影
            c = self.linears[i](c.reshape(b, ch, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embedding_dim, h, w)
            # DySample 动态放大到 512x512
            c = self.dysamples[i](c)
            outs.append(c)

        # 沿通道拼接
        out_c = torch.cat(outs, dim=1)

        # 送入耗时的 GBC 进行门控融合
        out_c = self.GBC_C(out_c)
        out_c = self.linear_fuse(out_c)
        out_c = self.dropout(out_c)

        # 预测掩膜
        logits = self.linear_pred_1(self.linear_pred(out_c))
        return logits
