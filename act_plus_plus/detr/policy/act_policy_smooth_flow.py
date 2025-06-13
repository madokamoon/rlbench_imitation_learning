import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import IPython
import hydra
import omegaconf
e = IPython.embed

import numpy as np
import copy

from util.visualize import visualize_multiple_attentions
class ACTPolicySmoothFLow(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.vq = args_override['vq']
        print(f'KL Weight {self.kl_weight}')
        self.attn_smooth_weight = args_override['attn_smooth_weight']
        print(f'Attention Smoothness Weight {self.attn_smooth_weight}')
        self.flow_attn_weight = args_override['flow_attn_weight']  # 默认权重
        print(f'Flow-Attention Loss Weight {self.flow_attn_weight}')
        
    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None, show_attn_weights=False, flow_data=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if show_attn_weights:
            curr_image = copy.deepcopy(image)
            
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries , attn_weights = self.model(qpos, image, env_state, actions, is_pad, vq_sample)

            # if show_attn_weights:
            #     import cv2
            #     from datetime import datetime
                
            #     # 保存当前训练状态
            #     training_mode = self.training
                
            #     # 临时切换到评估模式并禁用梯度
            #     self.eval()
            #     with torch.no_grad():
            #         # 重新运行以获取更稳定的注意力权重
            #         _, _, _, _, _, eval_attn_weights = self.model(qpos, image, env_state, actions, is_pad, vq_sample)
            #         final_image = visualize_multiple_attentions(curr_image, attn_weights, num_queries=10)
                
            #     # 恢复原来的训练模式
            #     if training_mode:
            #         self.train()
                    
            #     # 保存图像
            #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            #     filename = f"{timestamp}.png"
            #     os.makedirs("attention_vis/training", exist_ok=True)
            #     filepath = os.path.join("attention_vis/training", filename)
            #     cv2.imwrite(filepath, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            #     print(f"已保存多查询注意力可视化: {filepath}")

            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]

            # 计算并添加注意力平滑损失
            attn_smooth_loss = compute_masked_attention_smoothness_loss(attn_weights, is_pad)
            loss_dict['attn_smooth'] = attn_smooth_loss

            # 计算并添加光流损失
            flow_attn_loss, flow_blocks = compute_flow_attention_loss(attn_weights, flow_data, is_pad)
            loss_dict['flow_attn'] = flow_attn_loss

            # loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + flow_attn_loss * self.flow_attn_weight
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['attn_smooth'] * self.attn_smooth_weight + flow_attn_loss * self.flow_attn_weight

            print(f"Losses: {loss_dict}")
            
            if show_attn_weights:
                import cv2
                import os
                final_image = visualize_multiple_attentions(curr_image, attn_weights, num_queries=10,flow_map=flow_blocks)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{timestamp}.png"
                os.makedirs("attention_vis/training", exist_ok=True)
                filepath = os.path.join("attention_vis/training", filename)
                cv2.imwrite(filepath, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                print(f"已保存多查询注意力可视化: {filepath}")

                # cv2.imwrite("attention_vis/show_attention.png", cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                # print("已保存多查询注意力可视化: attention_vis/show_attention.png")
                # input("Press Enter to continue...")

            return loss_dict
        else: # inference time
            a_hat, is_pad_hat,  (mu, logvar), probs, binaries, attn_weights = self.model(qpos, image, env_state, vq_sample=vq_sample)
            if show_attn_weights:
                import cv2
                import os

                os.makedirs("attention_vis", exist_ok=True)
                for layer_idx in range(0, 7):
                    final_image = visualize_multiple_attentions(curr_image, attn_weights,layer_idx=layer_idx,num_queries=10)
                    filename = f"attention_layer_{abs(layer_idx)}.png"
                    filepath = os.path.join("attention_vis", filename)
                    cv2.imwrite(filepath, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                    print(f"已保存第{abs(layer_idx)}层的多查询注意力可视化: {filepath}")
                
                input("Press Enter to continue...")
            return a_hat

    def forward_pass(self, data):
        image_data, qpos_data, action_data, is_pad, flow_data = data
        image_data, qpos_data, action_data, is_pad, flow_data = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), flow_data.cuda()

        return self(qpos_data, image_data, action_data, is_pad, flow_data=flow_data)

    def configure_optimizers(self):
            return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def build_ACT_model_and_optimizer(args):

    build_model_config = omegaconf.DictConfig({
        "_target_": "act_plus_plus.detr.models." + args.model_name + ".build",
        'args': args
    })

    model = hydra.utils.call(build_model_config)
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

def make_policy(policy_config):
    # 创建策略模型
    policy = ACTPolicySmoothFLow(policy_config)
    return policy


def compute_attention_smoothness_loss(attn_weights, layer_idx=-1):
    """计算相邻动作之间的注意力平滑损失"""
    layer_attn = attn_weights[layer_idx]  # [batch_size, chunk_size, 902]
    
    # 使用从0到chunk_size-2的切片和从1到chunk_size-1的切片
    diff = layer_attn[:, 1:] - layer_attn[:, :-1]  # [batch_size, chunk_size-1, 902]
    
    # 计算L1损失
    smoothness_loss = torch.abs(diff).mean()
    
    return smoothness_loss

# 考虑填充掩码的改进版本
def compute_masked_attention_smoothness_loss(attn_weights, is_pad, layer_idx=-1):
    layer_attn = attn_weights[layer_idx]

    # 去除填充部分
    valid_mask = ~is_pad.unsqueeze(-1)  # [batch_size, chunk_size, 1]
    valid_pairs = valid_mask[:, :-1] & valid_mask[:, 1:]  # 相邻两个都有效
    
    diff = layer_attn[:, 1:] - layer_attn[:, :-1]
    masked_diff = diff * valid_pairs
    
    valid_count = valid_pairs.sum() + 1e-8  # 避免除零
    smoothness_loss = torch.abs(masked_diff).sum() / valid_count
    
    return smoothness_loss

def compute_flow_attention_loss(attn_weights, flow_data, is_pad=None, layer_idx=-1):
    """
    计算注意力图与光流数据之间的L1损失，排除填充动作
    
    Args:
        attn_weights: 注意力权重 [解码器层数, batch_size, chunk_size, token_num]
        flow_data: 光流数据 [batch, cameras, channels, height, width]
        is_pad: 填充掩码 [batch_size, chunk_size]
        layer_idx: 使用的注意力层索引，默认为最后一层
        
    Returns:
        torch.Tensor: L1损失值
        torch.Tensor: 平均注意力图 [batch_size, 1, total_feats]
    """
    
    # 获取特定层的注意力权重
    layer_attn = attn_weights[layer_idx]  # [batch_size, chunk_size, token_num]
    batch_size = layer_attn.shape[0]
    
    # 提取相机和图像尺寸信息
    num_cameras = flow_data.shape[1]
    height, width = flow_data.shape[3], flow_data.shape[4]
    
    # 确定特征图尺寸
    feat_h = 15
    feat_w = 20
    total_feats = feat_h * feat_w * num_cameras
    
    # 1. 处理注意力权重 - 跳过前两个非图像token
    query_attn = layer_attn[:, :, 2:2+total_feats]  # 只取需要的特征数量
    # 直接在原始形状上计算掩码平均值
    if is_pad is not None:
        # 创建有效掩码
        valid_mask = ~is_pad  # [batch_size, chunk_size]
        # 扩展掩码以匹配注意力权重的形状
        expanded_mask = valid_mask.unsqueeze(-1).expand_as(query_attn)  # [batch_size, chunk_size, total_feats]
        
        # 应用掩码并计算平均值
        masked_attn = query_attn * expanded_mask
        valid_counts = valid_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8  # 避免除零
        avg_attention_map = masked_attn.sum(dim=1, keepdim=True) / valid_counts  # [batch_size, 1, total_feats]
    else:
        # 如果没有提供掩码，直接计算平均值
        avg_attention_map = query_attn.mean(dim=1, keepdim=True)  # [batch_size, 1, total_feats]
    
    # 2. 处理flow_data - 先横向拼接相机数据，然后分块平均
    flow_first_channel = flow_data[:, :, 0]  # [batch, cameras, height, width]

    # 对每个batch和每个相机进行归一化处理
    normalized_flow = torch.zeros_like(flow_first_channel)
    for b in range(flow_first_channel.shape[0]):  # 遍历每个批次
        for c in range(flow_first_channel.shape[1]):  # 遍历每个相机
            # 获取当前batch和相机的光流图
            current_flow = flow_first_channel[b, c]
            
            # 计算最小值和最大值
            min_val = current_flow.min()
            max_val = current_flow.max()
            
            # 归一化到0-1范围，避免除零错误
            if max_val > min_val:
                # 正常归一化
                normalized_value = (current_flow - min_val) / (max_val - min_val)
                # print(f'c: {c}, min: {normalized_value.min()}, max: {normalized_value.max()}')
                # 对第二个相机(索引为1)进行反向归一化(1-值)
                if c == 61:
                    normalized_flow[b, c] = (1.0 - normalized_value)/2.0
                else:
                    normalized_flow[b, c] = normalized_value
            else:
                normalized_flow[b, c] = torch.zeros_like(current_flow)

    # 使用归一化后的光流数据
    flow_first_channel = normalized_flow

    flow_blocks = torch.zeros(batch_size, 1, total_feats, device=flow_data.device)
    
    # 计算块大小
    block_h = height // feat_h
    block_w = width // feat_w
    cam_feat_w = feat_w  # 每个相机的特征宽度应该是20
    
    # 首先，将所有相机的光流图拼接到一起（在内存中进行，不实际创建新张量）
    # 然后，按正确的格式提取块
    idx = 0
    for i in range(feat_h):  # 遍历行
        for j in range(feat_w * num_cameras):  # 遍历所有拼接后的列
            # 确定当前位置属于哪个相机
            cam_idx = j // cam_feat_w
            local_j = j % cam_feat_w  # 在当前相机内的列索引
            
            # 计算块在原始图像中的位置
            start_h = i * block_h
            end_h = min((i + 1) * block_h, height)
            start_w = local_j * block_w
            end_w = min((local_j + 1) * block_w, width)
            
            # 提取块并计算平均值
            block = flow_first_channel[:, cam_idx, start_h:end_h, start_w:end_w]
            flow_blocks[:, 0, idx] = torch.amax(block, dim=(1, 2))
            idx += 1
    
    # 对于可视化和调试
    # 可以将flow_blocks重塑为2D图像
    # flow_blocks_2d = flow_blocks.reshape(batch_size, 1, feat_h, feat_w * num_cameras)
    
    # 3. 计算L1损失
    # 归一化注意力图
    batch_size = avg_attention_map.shape[0]
    normalized_attention = torch.zeros_like(avg_attention_map)

    for b in range(batch_size):
        # 获取当前批次的注意力图
        curr_attn = avg_attention_map[b, 0]
        
        # 计算最小值和最大值
        attn_min = curr_attn.min()
        attn_max = curr_attn.max()
        
        # 归一化到0-1范围
        if attn_max > attn_min:
            normalized_attention[b, 0] = (curr_attn - attn_min) / (attn_max - attn_min)
        else:
            normalized_attention[b, 0] = torch.zeros_like(curr_attn)

    if True:
        # 直接计算全部区域的L1损失，不使用掩码过滤
        loss = F.l1_loss(normalized_attention, flow_blocks)
    else:
        # 创建掩码，只关注flow_blocks > 0.5的区域
        flow_mask = (flow_blocks > 0.5).float()

        # 应用掩码到注意力图和flow_blocks
        masked_attention = normalized_attention * flow_mask
        masked_flow_blocks = flow_blocks * flow_mask

        # 计算有效元素数量（避免除零错误）
        valid_elements = flow_mask.sum() + 1e-8

        # 只在掩码区域计算L1损失
        loss = torch.abs(masked_attention - masked_flow_blocks).sum() / valid_elements

    # 明确返回两个值: 损失和平均注意力图
    return loss, flow_blocks
