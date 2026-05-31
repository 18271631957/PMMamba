import math
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from .MSGM import MorphMixer

import torch
import torch.nn.functional as F



class MultiScaleCrossStripConv(nn.Module):
    def __init__(self, d_inner, bias=True):
        super().__init__()
        # 对半切分通道
        half_dim = d_inner // 2
        self.half_dim = half_dim


        # 1. 基础尺度分支 (Base): Kernel 9, Dilation 1
        # 计算 Padding: (9-1) * 1 / 2 = 4
        self.conv_h1 = nn.Conv2d(half_dim, half_dim, kernel_size=(1, 9), padding=(0, 4), groups=half_dim, bias=bias)
        self.conv_v1 = nn.Conv2d(half_dim, half_dim, kernel_size=(9, 1), padding=(4, 0), groups=half_dim, bias=bias)

        # 2. 宏观尺度分支 (Macro): Kernel 7, Dilation 3
        # 实际感受野 (RF): 1 + (7-1) * 3 = 19
        # 计算 Padding: (7-1) * 3 / 2 = 9
        self.conv_h2 = nn.Conv2d(half_dim, half_dim, kernel_size=(1, 13), padding=(0, 18), dilation=(1, 3), groups=half_dim, bias=bias) # SOTA 0.9244
        self.conv_v2 = nn.Conv2d(half_dim, half_dim, kernel_size=(13, 1), padding=(18, 0), dilation=(3, 1), groups=half_dim, bias=bias)

    def forward(self, x):
            # 沿通道维度对半切开: (B, 512, H, W) -> 两个 (B, 256, H, W)
            x1, x2 = torch.split(x, [self.half_dim, x.shape[1] - self.half_dim], dim=1)

            # 分别进行不同尺度的形态学感知
            out1 = self.conv_h1(x1) + self.conv_v1(x1)
            out2 = self.conv_h2(x2) + self.conv_v2(x2)

            # 原路拼接回去: (B, 512, H, W)
            return torch.cat([out1, out2], dim=1)




class MASS(nn.Module):
    def __init__(self, d_model=256, d_state=16, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_scale=1.0, dt_init_floor=1e-4, conv_size=7, bias=False, mass_conv_type='cs'):
        super().__init__()
        self.d_model = d_model  # embedding 256
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 2 * 256 = 512
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 256/16=16
        self.mass_conv_type = mass_conv_type

        self.silu = nn.SiLU()

        # 1. 输入投影 (升维) 翻倍投影：为了在后续 forward 中将特征对半切分为主干 (x) 和门控 (z) 两条分支
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1

        if self.mass_conv_type == 'cs':
            self.conv_spatial = MultiScaleCrossStripConv(self.d_inner)
            print("[MASS]: CS-Conv")
        elif self.mass_conv_type == 'normal':
            self.conv_spatial = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner, bias=bias)
            print("[MASS]: 3x3 DWConv")
        elif self.mass_conv_type == 'none':
            self.conv_spatial = nn.Identity()
            print("[MASS]: None")
        else:
            raise ValueError("conv_type only 'cs', 'normal' 或 'none'！")

        # 一次性生成Mamba参数 dt(步长), B(输入转移矩阵), C(输出转移矩阵)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)  # d_inner=2*256， dt_rank=16, d_state=16
        # 将低秩的 dt 重新投射回完整的 d_inner 维度
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 dt_proj 的权重保证模型收敛
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # 巧设 dt_proj 的偏置：强制符合对数分布，确保模型具备长短距离的感知能力
        # 计算对数空间下的上下边界
        log_dt_min = math.log(dt_min)
        log_dt_max = math.log(dt_max)
        rand_tensor = torch.rand(self.d_inner)
        # 在对数空间内进行均匀分布采样 公式映射: rand * (max - min) + min
        log_dt_rand = rand_tensor * (log_dt_max - log_dt_min) + log_dt_min
        # 通过指数函数还原到线性空间，并设置下限防止数值下溢
        dt = torch.exp(log_dt_rand).clamp(min=dt_init_floor)
        # Softplus的公式是 y=log(1+e^x)。 求它的反函数，结果应该是 x = log(e^y - 1), 等价变换x = y + log(1 - e^{-y})
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # 加锁，防止被 PyTorch 的全局初始化操作意外覆盖
        self.dt_proj.bias._no_reinit = True
        # 生成基础记忆衰减序列 如果 d_state 是 16，这里就会生成一个一维数组 [1.0, 2.0, 3.0, ..., 16.0] 在 Mamba模型中，这些数字代表了模型对不同信息的“遗忘速度”。
        base_decay_sequence = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        # 将一维序列扩展为二维矩阵。语法 "n -> d n" 的意思是，把刚才生成的长度为n=16的一维数组， 垂直向下复制 d (也就是 d_inner，比如 512) 次。
        A = repeat(base_decay_sequence, "n -> d n", d=self.d_inner).contiguous()
        A_log = torch.log(A)
        # (512, 16)
        self.A_log = nn.Parameter(A_log)
        # 禁止对A_log应用权重衰减（L2正则）
        self.A_log._no_weight_decay = True
        # (512,) 全1张量
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.B_bias = nn.Parameter(torch.zeros(1, self.d_state, 1))

    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape  # (B, L, d_model) -> 举例: (1, 4096, 256)
        H, W = hw_shape

        # (1, 4096, 256) -> (1, 4096, 512*2)
        xz = self.in_proj(x)
        # x: (1, 4096, 512), z: (1, 4096, 512) 对半劈开
        x, z = xz.chunk(2, dim=-1)
        # (1, 4096, 512) -> (1, 64, 64, 512) -> (1, 512, 64, 64)
        x_2d = x.reshape(batch_size, H, W, self.d_inner).permute(0, 3, 1, 2)
        # (1, 512, 64, 64) -> (1, 512, 64, 64)
        # x_2d = self.silu(self.bottconv(x_2d))
        # 十字方向感知，提取局部二维形态学特征，然后激活
        x_2d = self.conv_spatial(x_2d)
        x_2d = self.silu(x_2d)

        # (1, 512, 64, 64) -> (1, 64, 64, 512) -> (1, 4096, 512)
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, self.d_inner)

        # (1, 4096, 2*256) -> (1, 4096, 16+16*2)
        x_dbl = self.x_proj(x_conv)
        # (1, 4096, 16+16*2) -> (1, 4096, 16) + (1, 4096, 16) + (1, 4096, 16)
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # (1, 4096, 16) -> (1, 4096, 2*256)
        dt = self.dt_proj(dt)

        # 2. 【修复版：形态学步长调制】
        # 取 x_2d 的通道平均值 (B, 1, H, W)，然后 reshape 成 (B, L, 1) -> 即 (1, 4096, 1)
        morpho_gate = torch.sigmoid(x_2d.mean(dim=1).view(batch_size, L, 1))

        # 3. 动态调制！(1, 4096, 512) * (1, 4096, 1) -> 完美广播，不会报错！
        # 物理意义：裂缝处 dt 保持原样，背景处 dt 衰减接近 0，强制 Mamba 忽略背景。
        dt = dt * morpho_gate

        dt = dt.permute(0, 2, 1).contiguous()  # (1, 4096, 512) -> (1, 512, 4096)
        B_mat = B_mat.permute(0, 2, 1).contiguous()  # (1, 4096, 16) -> (1, 16, 4096)
        C_mat = C_mat.permute(0, 2, 1).contiguous()  # (1, 4096, 16) -> (1, 16, 4096)


        A = -torch.exp(self.A_log.float())

        # 直接把原始特征展平，死板扫描！
        x_dynamic_1d = x_2d.view(batch_size, self.d_inner, L).contiguous()
        B_with_bias = (B_mat + self.B_bias).contiguous()
        # Mamba扫描
        scan_out = selective_scan_fn(
            x_dynamic_1d,  # (1, 512, 4096)
            dt,  # 步长 (1, 512, 4096)
            A,  # 记忆矩阵 (512, 16)
            B_with_bias,  # 带有方向偏置的B矩阵 (B, 16, 4096)
            C_mat,  # C 矩阵 (1, 16, 4096)
            self.D.float(),  # 残差跳跃连接 (512,)
            z=None,  # 门控留给外部自己做
            delta_bias=self.dt_proj.bias.float(),  # (512,)
            delta_softplus=True,  # 保证步长为正数
            return_last_state=False  # 不需要返回最终状态
        )
        # (1, 512, 4096) -> (1, 4096, 512)
        y_restored = scan_out.permute(0, 2, 1).contiguous()

        # (1, 4096, 512) * (1, 4096, 512) -> (1, 4096, 512)
        y = y_restored * self.silu(z)
        # (1, 4096, 512) -> (1, 4096, 256)
        out = self.out_proj(y)
        return out


class MASS_Layer(nn.Module):
    def __init__(self, embed_dims, drop_path_rate, layer_conv_times, use_layer_conv, use_res, use_synergy, mamba_cfg):
        super(MASS_Layer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})  # 256

        self.layer_conv_times = layer_conv_times
        self.use_layer_conv = use_layer_conv
        self.use_res = use_res

        self.layer_norm = nn.LayerNorm(embed_dims)
        self.dsass = MASS(**mamba_cfg)
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        if self.use_res:
            # 简单的线性映射层，用于最后的残差处理
            self.linear_256 = nn.Linear(in_features=embed_dims, out_features=embed_dims, bias=True)
        # GroupNorm (分组归一化)，相比 BatchNorm 更适合小的 Batch Size
        self.GN_256 = nn.GroupNorm(num_channels=embed_dims, num_groups=16)
        # 专门用来补充裂缝局部的二维几何纹理信息
        self.GBC_C = MorphMixer(embed_dims)

        # ==================== 【SegMAN 绝杀：滑动局部协同】 ====================
        self.use_synergy = use_synergy
        print(f"[MASS_Layer] use_synergy: {self.use_synergy}")
        if self.use_synergy:
            # 只有开启开关时，才初始化这两个参数，绝不多占一丁点显存
            self.local_aggregator = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, groups=embed_dims, bias=False),
                nn.SiLU()
            )
            self.synergy_alpha = nn.Parameter(torch.tensor([0.1]))

        # ================================================================

        print("layer_conv_times", layer_conv_times, "use_synergy", use_synergy, "use_layer_conv", use_layer_conv, "use_res", use_res)

    def forward(self, x, hw_shape):  # 输入(1, 4096, 256), (64, 64)
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        # (1, 4096, 256) -> (1, 64, 64, 256) -> (1, 256, 64, 64)  <-- 注释修正
        x_2d_input = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.use_layer_conv:
            for i in range(self.layer_conv_times):
                x_2d_input = self.GBC_C(x_2d_input)

        # (1, 256, 64, 64) -> (1, 64, 64, 256) -> (1, 4096, 256)
        x_flat = x_2d_input.permute(0, 2, 3, 1).reshape(B, H * W, C)

        mamba_in = self.layer_norm(x_flat)
        mamba_out = self.dsass(mamba_in, hw_shape)
        mixed_x = self.drop_path(mamba_out)

        b, l, c = mixed_x.shape
        h = w = int(math.sqrt(l))

        # 动态协同融合
        if self.use_synergy:
            # 开启时：提取滑动局部高保真特征并协同融合
            local_detail_2d = self.local_aggregator(x_2d_input)
            local_detail_1d = local_detail_2d.permute(0, 2, 3, 1).reshape(b, l, c)

            # 可解释
            self.debug_local_feat = local_detail_2d.detach()  # Conv 提取的局部锐利特征
            self.debug_mamba_feat = mixed_x.permute(0, 2, 1).reshape(b, c, h, w).detach()  # Mamba 提取的全局连通特征

            # mixed_x = mixed_x + self.synergy_alpha * local_detail_1d
            mixed_x = mixed_x + self.synergy_alpha * local_detail_1d  # DeepCrack上没用

        # (1, 4096, 256) -> (1, 256, 4096) -> (1, 256, 64, 64)
        mixed_x_2d = mixed_x.permute(0, 2, 1).reshape(b, c, h, w)
        x_2d = x_flat.permute(0, 2, 1).reshape(b, c, h, w)

        mixed_x_fused = x_2d + mixed_x_2d

        # (1, 256, 64, 64) -> (1, 256, 4096) -> (1, 4096, 256)
        mixed_x_out = self.GN_256(mixed_x_fused).reshape(b, c, h * w).permute(0, 2, 1)

        if self.use_res:
            mixed_x_for_res = self.GN_256(mixed_x_out.permute(0, 2, 1)).permute(0, 2, 1)
            mixed_x_res = self.linear_256(mixed_x_for_res)
            return mixed_x_out + mixed_x_res
        else:
            return mixed_x_out
