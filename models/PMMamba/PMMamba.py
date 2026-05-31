import torch
import torch.nn as nn

from .layer.MFA import MFA
from .layer.decoder.Ham import HamDecoderForMamba
from .layer.decoder.MFS import MFS
from .layer.decoder.SegFormer_Decoder import SegFormerDecoderForMamba
from .layer.decoder.UNet_Decoder import UNetDecoderForMamba
from .layer.morph_encoder import MorphEncoder

import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


class LOSS_bce_dice(nn.Module):
    def __init__(self, args, model):
        super(LOSS_bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        self.args = args

    def forward(self, y_pred, y_true, epoch=None):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.args.weight_bce * bce + self.args.weight_dice * dice


def criterion(args, model):
    return LOSS_bce_dice(args, model)


class Model(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        if not hasattr(args, 'out_indices') or args.out_indices is None:
            out_indices = tuple(range(args.num_layers))
        else:
            out_indices = args.out_indices
        print(args.num_layers, out_indices)
        self.backbone = MorphEncoder(args=args, embed_dims=args.encoder_out_dim, num_layers=args.num_layers, out_indices=out_indices, drop_path_rate=0.2)

        backbone_out_channels = args.encoder_out_dim

        if args.decoder_type == "MFA":
            self.decoder = MFA(args, embedding_dim=args.mfa_linear_out_dim, out_layer_num=len(out_indices), encoder_out_dim=args.encoder_out_dim)
        elif args.decoder_type == "MFS":
            self.decoder = MFS(MASSLayer_out_dim=backbone_out_channels, embedding_dim=8, num_layers=len(out_indices))
        elif args.decoder_type == "UNet":
            self.decoder = UNetDecoderForMamba(MASSLayer_out_dim=backbone_out_channels, num_classes=1)
        elif args.decoder_type == "SegFormer":
            self.decoder = SegFormerDecoderForMamba(in_channels=backbone_out_channels, decoder_dim=256, num_classes=1, num_layers=len(out_indices))
        elif args.decoder_type == "Ham":
            self.decoder = HamDecoderForMamba(in_channels=backbone_out_channels, num_layers=len(out_indices), num_classes=1)
        else:
            raise NotImplementedError(f"不支持的解码器类型: {args.decoder_type}")

    def forward(self, samples):
        outs_SAVSS = self.backbone(samples)
        out = self.decoder(outs_SAVSS)
        return out
