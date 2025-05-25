import torch
from torch import nn
from segment_anything import sam_model_registry

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from models.position_encoding import build_position_encoding

import IPython
e = IPython.embed

class SAMEncoder(nn.Module):
    def __init__(self, sam_checkpoint, model_type="vit_h", sam_feature_dim = 256, device="cuda"):
        super().__init__()
        # 只加载图像编码器部分
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder
        
        # 冻结SAM编码器参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        self.device = device
        self.num_channels = sam_feature_dim
        self.image_encoder.to(device)
    
    def forward(self, image):
        """
            输入图像预处理并使用SAM图像编码器生成嵌入表示。

            参数:
                image (torch.Tensor): 输入图像张量，形状应为 [B, C, H, W]，
                                      并且 H, W 接近或等于 1024。

            返回:
                torch.Tensor: 图像嵌入向量
            """
        # 检查当前设备
        device = image.device

        # 将图像缩放到 1024x1024（保持宽高比）
        resized_image = F.interpolate(image, size=(1024, 1024), mode='bilinear', align_corners=False)

        with torch.no_grad():  # 确保不计算梯度
            image_embedding = self.image_encoder(resized_image)

        return image_embedding

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)



    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        # for name, x in xs.items():
        out.append(xs)
        # position encoding
        pos.append(self[1](xs).to(xs.dtype))

        return out, pos

def build_sam_encoder(args):
    model = SAMEncoder(sam_checkpoint=args.sam.sam_checkpoint,
                       model_type=args.sam.model_type,
                       sam_feature_dim=args.sam.sam_feature_dim)
    return model


def build_sam_backbone(args):
    position_embedding = build_position_encoding(args)
    # model = Joiner(sam, position_embedding)
    # model.num_channels = sam.num_channels
    return position_embedding
