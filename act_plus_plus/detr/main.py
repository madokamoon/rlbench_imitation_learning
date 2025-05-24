# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # act修改代码优化 新增参数 --config 因为 data_sampler.py 中需要用到
    parser.add_argument('--config', type=str, default=None)

    
    # SAM相关参数
    parser.add_argument('--use_sam', action='store_true', default=True, 
                        help='使用SAM视觉编码器')
    parser.add_argument('--sam_checkpoint', type=str, 
                        default='/home/madoka/python/sam_vit_h_4b8939.pth',
                        help='SAM模型检查点路径')
    parser.add_argument('--sam_type', type=str, default='vit_h', 
                        choices=['vit_h', 'vit_l', 'vit_b'], 
                        help='SAM编码器类型')
        
    return parser


def build_ACT_model_and_optimizer(args):
    import types
    args = types.SimpleNamespace(**args)
    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

