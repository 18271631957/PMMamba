import torch
import torch.nn as nn
from .DySample import DySample
import torch.nn.functional as F


class MFA(nn.Module):
    """
        Morphology-driven Feature Aggregator 形态学驱动特征聚合器
        [添加了消融实验专用的双开关: use_dysample 和 use_sdf]
    """

    def __init__(self, args, out_layer_num, embedding_dim=8, encoder_out_dim=256):
        super(MFA, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_dysample = args.use_dysample
        print("use_dysample", self.use_dysample)
        # 使用 nn.ModuleList 动态创建投影层
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(encoder_out_dim, embedding_dim, kernel_size=1)
            for _ in range(out_layer_num)
        ])
        # 动态上采样
        if self.use_dysample:
            self.dy_upsample_8x = DySample(embedding_dim, scale=8, dyscope=args.use_dysample)

        # 融合
        # self.vanilla_fusion = nn.Sequential(
        #     nn.Conv2d(embedding_dim * out_layer_num, embedding_dim * out_layer_num, kernel_size=3, padding=1),
        #     nn.GroupNorm(out_layer_num, embedding_dim * out_layer_num),
        #     nn.GELU()
        # )
        total_dim = embedding_dim * out_layer_num  # 64
        self.vanilla_fusion = nn.Sequential(
            # 第一层：深度可分离卷积，提取跨层空间关联
            nn.Conv2d(total_dim, total_dim, kernel_size=3, padding=1, groups=total_dim, bias=False),
            nn.GroupNorm(out_layer_num, total_dim),
            nn.GELU(),
            # 第二层：1x1 卷积实现通道间的特征重组 (关键步骤)
            nn.Conv2d(total_dim, total_dim * 2, kernel_size=1, bias=False),
            nn.GELU(),
            # 第三层：压缩回原维度并引入残差连接思路
            nn.Conv2d(total_dim * 2, total_dim, kernel_size=1, bias=False)
        )

        self.linear_pred = nn.Conv2d(embedding_dim * out_layer_num, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        upsampled_features = []

        # 循环处理每一层
        for i, layer_feat in enumerate(inputs):
            low_dim_feat = self.linear_layers[i](layer_feat)  # 投影降维

            # ======== 【执行退化 1】 ========
            if self.use_dysample:
                up_feat = self.dy_upsample_8x(low_dim_feat)  # 动态上采样 (神装)
            else:
                up_feat = F.interpolate(low_dim_feat, scale_factor=8, mode='bilinear', align_corners=False)  # 双线性插值 (素车)

            upsampled_features.append(up_feat)

        # 一次性拼接所有特征
        concat_feat = torch.cat(upsampled_features, dim=1)


        enriched_feat = self.vanilla_fusion(concat_feat)  # 盲目融合 (素车)



        out_c = self.dropout(enriched_feat)
        x = self.linear_pred(out_c)

        return x
