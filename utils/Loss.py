import torch
import torch.nn as nn
import torch.nn.functional as F

# 二分类交叉熵损失
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def forward(self, logits, target): # logits = 模型原始输出（没经过 sigmoid /softmax，范围任意：-∞ ~ +∞）
        """
            logits: 模型原始输出（没有经过sigmoid）
            target: 真实标签（0或1）
        """
        # sigmoid，把logits转成 0~1 概率
        prob = torch.sigmoid(logits)
        # 第二步：逐像素计算 BCE 损失（核心公式）
        # 公式：- [ y * log(p) + (1 - y) * log(1 - p) ]
        bce_loss = - (target * torch.log(prob + 1e-8) + (1 - target) * torch.log(1 - prob + 1e-8))
        # 整个batch里的所有像素全部一起平均
        return bce_loss.mean()

# Dice损失
class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        """
        预测x：        真实y：
            [[0.9, 0.1],  [[1, 0],
             [0.2, 0.8]]   [0, 1]]
        第一步：逐像素相乘
            0.9*1  0.1*0   →   0.9    0
            0.2*0  0.8*1   →    0    0.8
        第二步：.sum((-2, -1))对最后两维（行、列）全部加起来
            tp = 0.9 + 0 + 0 + 0.8 = 1.7
        """
        tp = (x * y).sum(self.dims) # 猜对的裂缝
        fp = (x * (1 - y)).sum(self.dims) # 误报的裂缝
        fn = ((1 - x) * y).sum(self.dims) # 漏掉的裂缝
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth) # 计算Dice系数
        dc = dc.mean() # 对batch内所有样本取平均

        return 1 - dc