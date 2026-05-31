from types import SimpleNamespace
from main import main


# 1. 定义默认基础参数（复用原有默认值）
def get_default_args():
    # 损失函数参数
    loss_args = SimpleNamespace(
        weight_bce=1,
        weight_dice=1,
        weight_cldice=None,
    )

    # 数据集相关基础参数（dataset_path后续会被覆盖）
    data_args = SimpleNamespace(
        dataset_path="./data/CrackMap",
        batch_size_train=4,
        batch_size_test=1,
        num_threads=1,
        load_width=512,
        load_height=512,
        phase='train'
    )

    # 优化器/学习率参数
    optim_args = SimpleNamespace(
        lr_scheduler='PolyLR',
        lr=5e-4,
        min_lr=1e-6,
        weight_decay=0.01,
        epochs=50,
        start_epoch=0,
        lr_drop=30,
        sgd=False,
        use_dysample=True,
        use_layer_conv=True,
        use_res=True,
        use_synergy=True,
        mass_conv_type='cs',  # cs/normal/none
        num_layers=4,
        layer_conv_times=(2, 2, 2, 2),
        decoder_type='MFA'  # UNet/Ham/SegFormer/MFS
    )

    # 通用基础参数（model_name后续会被覆盖）
    common_args = SimpleNamespace(
        output_dir='./checkpoints/weights',
        device='cuda',
        seed=42,
        model_name='XXX',
        dsass_off=1e-4,
        warmup_epochs=0,
        sava_all_checkpoint=False,
    )

    # 合并所有默认参数
    args = SimpleNamespace(
        **vars(loss_args),
        **vars(data_args),
        **vars(optim_args),
        **vars(common_args)
    )
    return args


train_tasks = [
    {"model_name": "PMMamba", "dataset_path": "./data/CrackMap", "encoder_out_dim": 128, "mfa_linear_out_dim": 16},

]

# 3. 遍历执行每个训练任务
if __name__ == "__main__":
    for idx, task in enumerate(train_tasks):
        args = get_default_args()
        for key, value in task.items():
            setattr(args, key, value)

        # 打印任务信息（便于日志追踪）
        print(f"\n========== 执行第 {idx + 1}/{len(train_tasks)} 个训练任务 ==========")
        print(f"模型名称: {args.model_name}")
        print(f"数据集路径: {args.dataset_path}")
        print(f"调度器: {args.lr_scheduler} | 学习率: {args.lr}")
        print("=" * 50)

        main(args)
