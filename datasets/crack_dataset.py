import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class CrackDataset(Dataset):
    def __init__(self, args, flag='train'):
        self.args = args
        self.flag = flag
        self.exts = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
        self.img_dir = os.path.join(args.dataset_path, f'{flag}_img')
        self.lab_dir = os.path.join(args.dataset_path, f'{flag}_lab')

        self.img_paths = self._make_dataset(self.img_dir)  # 递归搜索图片
        if len(self.img_paths) == 0:
            raise RuntimeError(f"在 {self.img_dir} 中没找到图片，请检查路径。")

        # 1.设置归一化参数
        if args.model_name == 'Crackmer':
            norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    def _make_dataset(self, dir_path):
        images = []  # 递归遍历文件夹，获取所有符合图片后缀的文件绝对路径。
        if not os.path.isdir(dir_path):
            return images
        # 2. os.walk(dir_path) 会递归遍历该文件夹下的所有子文件夹
        # root: 当前正在遍历的文件夹路径 _: 当前文件夹下的子文件夹列表（这里用不到，所以用下划线忽略）
        # fnames: 当前文件夹下的所有文件名列表
        for root, _, fnames in sorted(os.walk(dir_path)):
            for fname in fnames:
                # 3. 检查文件名是否以指定的图片后缀结尾
                # any(...) 会检查括号内的条件，只要有一个为 True，结果就是 True
                # 这行代码的意思是：如果文件名以 .jpg 或 .png 等任意一个后缀结尾
                if any(fname.endswith(ext) for ext in self.exts):
                    # 将文件夹路径和文件名拼接成完整的绝对路径 例如：'datasets/DeepCrack/train_img' + '001.jpg' -> 'datasets/DeepCrack/train_img/001.jpg'
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __getitem__(self, index):
        """
            根据索引 index 获取单张图片和对应标签，并完成所有预处理操作。
        """
        img_path = self.img_paths[index]  # 按 index 取出当前这张图的绝对路径
        # "data/DeepCrack/train_img/001.jpg" -> 001.jpg -> 001
        file_name = os.path.basename(img_path).rsplit('.', 1)[0]
        lab_path = os.path.join(self.lab_dir, file_name + '.png')

        # cv2.imread 读取图片。cv2.IMREAD_UNCHANGED 表示原样读取（包含所有通道，不随意压缩）
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # OpenCV 默认读取的颜色通道是 BGR（蓝绿红），但深度学习里通常用 RGB（红绿蓝）
        # 所以必须用 cv2.cvtColor 把它翻转过来，否则模型学到的颜色是反的
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 读取标签图片
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        # 容错处理：万一这张图没有对应的标签文件，就生成一个全黑（全是背景）的假标签，防止训练中断
        if lab is None:
            # np.zeros 生成全 0 矩阵，大小和原图的高、宽一致。uint8 是图像常用的 8 位无符号整数类型
            lab = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # 如果读进来的标签是彩色的（3个通道），把它转成单通道灰度图
        # 语义分割的标签通常是单通道的（比如 0 是背景，1 或者 255 是裂缝）
        elif len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)


        # 测试阶段中，常规模型的处理
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            # 缩放原图：使用三次插值 (INTER_CUBIC)，这样缩放出来的图片边缘更平滑，质量好
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            # 缩放标签：必须使用最近邻插值 (INTER_NEAREST)
            # 因为标签的值代表类别（如 0 和 255）。如果用平滑插值，会产生 128 这种没有意义的小数类，导致报错！
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)

        # 标签阈值二值化 深度学习中，分割标签的值必须是确切的类别索引（如 0 代表背景，1 代表裂缝）
        if self.args.dataset_path.split('/')[-1] in ['CrackTree260', 'CrackForest']:
            # 模拟原作者的加粗操作：只要像素值大于 0（不是纯黑），统统被强制设为 1（裂缝）
            # 这通常用来处理裂缝边缘模糊的情况，会让网络更容易检测到极细的裂缝
            _, lab = cv2.threshold(lab, 0, 1, cv2.THRESH_BINARY)
        else:
            _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
            _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        # 格式转换：转为 PyTorch 需要的 Tensor
        # 原图转换：因为之前定义的 transforms 期望输入 PIL 格式的图片，
        # 用Image.fromarray 把 numpy 数组转成 PIL 对象，再进行 ToTensor 和归一化
        img_tensor = self.img_transforms(Image.fromarray(img))
        # 标签转换：torch.from_numpy把 numpy 数组直接变成 torch 张量
        # .long(): 标签在计算交叉熵损失等 Loss 时，数据类型必须是 64 位整型 (LongTensor)
        # .unsqueeze(0): (H, W) -> (1, H, W)，表示这是 1 个通道的数据
        lab_tensor = torch.from_numpy(lab).long().unsqueeze(0)

        return {
            'image': img_tensor,
            'label': lab_tensor,
            'A_paths': img_path,
            'B_paths': lab_path
        }

    def __len__(self):
        return len(self.img_paths)


def get_data_loader(args, flag='train'):
    dataset = CrackDataset(args, flag=flag)
    is_train = (flag == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_train if is_train else args.batch_size_test,
        shuffle=is_train,
        num_workers=int(args.num_threads),
        pin_memory=True
    )
    return dataloader
