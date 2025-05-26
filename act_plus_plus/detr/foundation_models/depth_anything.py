from foundation_models.depth_anything_v2.dpt import DepthAnythingV2
import torch
import torch.nn.functional as F
from torch import nn
import IPython
import torch.nn.functional as F
from torchvision.transforms import Compose
from foundation_models.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2

e = IPython.embed


class DepthAnything(nn.Module):
    def __init__(self, args):
        super().__init__()
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        # 只加载depth_anything
        checkpoint = args["checkpoint"]
        model_type = args["model_type"]
        self.model = DepthAnythingV2(**model_configs[model_type])
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        # 冻结 depth_anything 参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(DEVICE).eval()

        self.input_size = 518 # 预训练模型需要固定的图片输入尺寸

    def forward(self, image: torch.Tensor):
        h, w = image.shape[-2], image.shape[-1]
        with torch.no_grad():  # 确保不计算梯度
            image = F.interpolate(image, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
            # image = self.transform({'image': image.cpu().numpy()})['image']
            depth = self.model.forward(image)
            depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)
        return depth

def build_depth_anything(args):
    model = DepthAnything(args["depth_anything"])
    return model
