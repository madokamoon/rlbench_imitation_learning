import torch
from torch import nn
from segment_anything import sam_model_registry

class SAMEncoder(nn.Module):
    def __init__(self, sam_checkpoint, model_type="vit_b", device="cuda"):
        super().__init__()
        # 只加载图像编码器部分
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder
        
        # 冻结SAM编码器参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        self.device = device
        self.image_encoder.to(device)
    
    def forward(self, image):
        # SAM编码器输入预处理
        # 注意SAM输入尺寸要求可能与原始模型不同，需要调整
        with torch.no_grad():  # 确保不计算梯度
            image_embedding = self.image_encoder(image)
        return image_embedding