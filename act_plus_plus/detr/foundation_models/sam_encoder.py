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
    def __init__(self, args):
        super().__init__()
        # 只加载图像编码器部分
        sam_checkpoint = args["sam_checkpoint"]
        model_type = args["model_type"]
        sam_feature_dim = args["sam_feature_dim"]
        sam_2d_dim = args["sam_2d_dim"]
        output_2d_dim_1 = args["output_2d_dim"]["dim_1"]
        output_2d_dim_0 = args["output_2d_dim"]["dim_0"]
        self.sam_2d_dim = sam_2d_dim
        self.output_2d_dim_1 = output_2d_dim_1
        self.output_2d_dim_0 = output_2d_dim_0
        # 将数据维度缩小，防止爆显存
        self.output_proj = nn.Linear(sam_2d_dim * sam_2d_dim, output_2d_dim_1 * output_2d_dim_0)
        # unconfig
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder
        
        # 冻结SAM编码器参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        self.device = device
        self.num_channels = sam_feature_dim
        self.image_encoder.to(device).eval()
    
    def forward(self, image: torch.Tensor):
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
        with torch.no_grad():  # 确保不计算梯度
            resized_image = F.interpolate(image, size=(1024, 1024), mode='bilinear', align_corners=False)
            image_embedding = self.image_encoder(resized_image)
        image_embedding = image_embedding.reshape([-1, self.sam_2d_dim * self.sam_2d_dim])
        # 4*256 64*64 -> 4*256 15*20
        image_embedding = self.output_proj(image_embedding)
        # 4*256 15*20 -> 4 256 15 20
        image_embedding = image_embedding.reshape([-1, self.num_channels, self.output_2d_dim_1, self.output_2d_dim_0])
        return image_embedding

def build_sam_encoder(args):
    model = SAMEncoder(args["sam"])
    return model
