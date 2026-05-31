import os

from datasets.crack_dataset import get_data_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from utils.evaluate_me import eval
from utils.logger import get_logger
from tqdm import tqdm
from mmengine.optim.scheduler.lr_scheduler import PolyLR
import torch.distributed as dist
import pandas as pd
from models.PMMamba import PMMamba


def get_rank():
    """获取当前进程/显卡的编号。单卡模式下默认为 0"""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """判断当前是否为主进程"""
    return get_rank() == 0


def log_model_param(logger, model):
    trained_parameters = []
    numel_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
            num_params = p.numel()
            numel_params = numel_params + num_params
            logger.info(f"Layer: {name}, Parameters: {num_params}")
    logger.info(f"All Parameters Num: {len(trained_parameters)}, All Numel:{numel_params}")
    return trained_parameters


def get_model(args):
    model_dict = {
        'PMMamba': PMMamba,

    }
    model_cls = model_dict[args.model_name]
    model = model_cls.Model(args)
    # 统一传参，内部自己决定用不用 model
    criterion = model_cls.criterion(args, model)
    return model, criterion


def main(args):
    checkpoints_path = "./checkpoints"
    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    process_folder_path = os.path.join(checkpoints_path, args.model_name, args.model_name + '_' + cur_time + '_' + dataset_name)
    Path(process_folder_path).mkdir(parents=True, exist_ok=True)

    log_train = get_logger(process_folder_path, 'train')
    log_test = get_logger(process_folder_path, 'test')
    log_eval = get_logger(process_folder_path, 'eval')

    log_train.info("args -> " + str(args))
    args.device = torch.device(args.device)

    # 设置随机种子
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建模型
    model, criterion = get_model(args)

    log_model_param(log_train, model)
    model.to(args.device)
    args.batch_size = args.batch_size_train

    train_loader = get_data_loader(args, flag='train')
    test_loader = get_data_loader(args, flag='test')
    log_train.info(f'The number of training images = {len(train_loader)}')

    params = model.parameters()
    # 模型优化器
    if args.sgd:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # 学习率迭代器
    if args.lr_scheduler == 'StepLR':
        # 阶梯式学习率：每经过 args.lr_drop 个 epoch，学习率乘以 gamma（默认0.1）
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosAWS':
        # T_0=30    第一个重启周期的长度（step/epoch）
        # T_mult=2  每个后续周期长度为上一周期的2倍（周期逐渐变长）
        # eta_min=1e-5  学习率下降的最小值，不会低于该值
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        # 多项式衰减学习率：学习率平滑缓慢下降至最小值
        # eta_min=args.min_lr  学习率最低值
        # begin=args.start_epoch  开始衰减的epoch
        # end=args.epochs  衰减结束的epoch（总训练轮数）
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)
    elif args.lr_scheduler == 'ReduceLR':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")
    output_dir = Path(process_folder_path) / 'weights'
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    max_mIoU = 0

    all_metrics_history = []  # 累计指标
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        criterion.train()

        # train
        total_loss = 0.0  # 用于计算整个 epoch 的平均 Loss
        for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"TRAIN Epoch {epoch}"):
            samples = data['image'].to(args.device)
            targets = data['label'].to(args.device)

            optimizer.zero_grad()
            output = model(samples)
            loss_final = criterion(output, targets.float(), epoch)

            loss_final.backward()
            optimizer.step()
            total_loss += loss_final.item()

        # 每个 Epoch 结束后，集中写入一次日志
        avg_loss = total_loss / len(train_loader)
        cur_lr = optimizer.param_groups[0]['lr']
        log_train.info(f"Time: {time.strftime('%Y%m%d_%H%M%S', time.localtime())} | Epoch: {epoch} | Avg Loss: {avg_loss} | LR: {cur_lr:}")

        # 学习率步进（修复 ReduceLROnPlateau 传参问题）
        if args.lr_scheduler == 'ReduceLR':
            lr_scheduler.step(avg_loss)  # ReduceLROnPlateau 需要传入监控指标（这里用训练平均损失）
        else:
            lr_scheduler.step()  # 其他调度器（StepLR/CosLR/PolyLR）直接step

        save_root = Path(process_folder_path) / 'results' / f'results_epoch_{epoch}'

        args.batch_size = args.batch_size_test

        # 在这之前初始化
        all_preds = []
        all_gts = []
        all_names = []
        all_losses = []  # 用于记录每个样本的 loss

        with torch.no_grad():
            model.eval()
            for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"TEST Epoch {epoch}"):
                x = data["image"].to(args.device)
                target = data["label"].to(dtype=torch.int64, device=args.device)

                out = model(x)
                loss = criterion(out, target.float(), epoch)

                target = target[0, 0, ...].cpu().numpy()  # (batch_size, channel_num, H, W) -> (H, W)
                out = out[0, 0, ...].cpu().numpy()  # (batch_size, channel_num, H, W) -> (H, W)

                target = 255 * (target / np.max(target))
                out = 255 * (out / np.max(out))
                root_name = data["A_paths"][0].split("/")[-1][0:-4]

                # --- 关键修改：存入内存列表 ---
                all_preds.append(out)
                all_gts.append(target)
                all_names.append(root_name)  # 假设 root_name 是 list
                all_losses.append(loss.item())  # 存下 loss 数值
                #
                # # 保存图片
                # lab_path = save_root / f"{root_name}_lab.png"
                # pre_path = save_root / f"{root_name}_pre.png"
                # cv2.imwrite(lab_path, target)
                # cv2.imwrite(pre_path, out)
                # h, w = target.shape[:2]
                # separator = np.ones((h, 10), dtype=np.uint8) * 255  # 生成 白色 分隔线（宽度10像素，可自己改）
                # concat_img = np.hstack((target, separator, out))  # 左右拼接：标签 | 白色分隔线 | 预测
                #
                # concat_path = save_root / "concat"
                # concat_path.mkdir(parents=True, exist_ok=True)
                # concat_path = concat_path / f"{str(loss.item())}_{root_name}.png"
                # cv2.imwrite(concat_path, concat_img)

        log_test.info("epoch " + str(epoch) + " test finish!")

        # 全内存
        metrics = eval(log_eval, all_preds, all_gts, epoch)
        # metrics = eval(log_eval, save_root, epoch)

        metrics['is_current_best'] = 0
        save_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        if (max_mIoU < metrics['mIoU']):
            max_mIoU = metrics['mIoU']
            metrics['is_current_best'] = 1
            # --- 拼接并保存最好结果 ---
            # save_root.mkdir(parents=True, exist_ok=True)
            # for pred, label, img_name, l_val in zip(all_preds, all_gts, all_names, all_losses):
            #     # lab_path = save_root / f"{img_name}_lab.png"
            #     # pre_path = save_root / f"{img_name}_pre.png"
            #     # cv2.imwrite(lab_path, label)
            #     # cv2.imwrite(pre_path, pred)
            #
            #     h, w = label.shape[:2]
            #     separator = np.ones((h, 10), dtype=np.uint8) * 255
            #     concat_img = np.hstack((label, separator, pred))
            #
            #     concat_path = save_root / "concat"
            #     concat_path.mkdir(parents=True, exist_ok=True)
            #     concat_path = concat_path / f"{l_val:.4f}_{img_name}.png"
            #     cv2.imwrite(concat_path, concat_img)
            # --------------------------
            if is_main_process():  # 保存最好的模型
                torch.save(save_state, output_dir / f'checkpoint_best.pth')
            log_eval.info(f"Update and save best model -> Epoch {epoch} with mIoU: {max_mIoU}")

        if args.sava_all_checkpoint:
            torch.save(save_state, output_dir / f'checkpoint_{epoch}.pth')

        # 累计所有epoch指标
        all_metrics_history.append(metrics)
        df_metrics = pd.DataFrame(all_metrics_history)
        # 保存为CSV
        df_metrics.to_csv(output_dir.parent / "eval_metrics_table.csv", index=False, float_format='%.4f')

    # 计算整个过程花了多少时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log_train.info('Process time {}'.format(total_time_str))
