from types import SimpleNamespace

from thop import profile
import torch


from models.PMMamba.PMMamba import Model as PMMamba

model_dict = {
    'PMMamba': PMMamba,
}

if __name__ == '__main__':
    common_args = SimpleNamespace(
        use_dysample=True,
        mass_conv_type='cs',
        use_layer_conv=True,
        use_res=True,
        use_synergy=True,
        decoder_type='MFA',
        num_layers=4,
        layer_conv_times=(2, 2, 2, 2),
        encoder_out_dim=128,
        mfa_linear_out_dim=16
    )

    # 创建模型
    model = PMMamba(common_args)
    model.to('cuda')

    input = torch.randn(1, 3, 512, 512)
    samples = input.to('cuda')

    flops, params = profile(model, (samples,))
    print("flops(G):", flops, "params(M):", params)
    print("flops(G):", flops / 1e9, "params(M):", params / 1e6)

    # PyTorch 默认创建出来的模型使用 float32 精度，每个参数占 4 个字节 (Bytes)
    size_in_bytes = params * 4  # 计算总字节数
    size_in_kb = size_in_bytes / 1024  # 转换为 KB
    size_in_mb = size_in_kb / 1024  # 转换为 MB
    print(f"Size(MB): {size_in_mb:.2f}")

    print(f"{(params / 1e6):.2f}", f"{(flops / 1e9):.2f}", f"{size_in_mb:.2f}")
