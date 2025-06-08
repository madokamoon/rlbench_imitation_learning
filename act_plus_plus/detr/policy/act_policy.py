import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import os,sys,copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import IPython
import hydra
import omegaconf
e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.vq = args_override['vq']
        print(f'KL Weight {self.kl_weight}')


    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None, show_attn_weights=False):
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
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, is_pad_hat, vae_out, probs, binaries, attn_weights = self.model(qpos, image, env_state, vq_sample=vq_sample)
            if show_attn_weights:
                visualize_multiple_attentions(curr_image, attn_weights, num_queries=10)
                input("Press Enter to continue...")
            return a_hat

    def forward_pass(self, data):
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()

        return self(qpos_data, image_data, action_data, is_pad,)

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
    policy = ACTPolicy(policy_config)
    return policy

def visualize_multiple_attentions(image, attn_weights, num_queries=15, layer_idx=-1):
    """将原始图像和多个查询的注意力热图竖向拼接为一张长条图像，中间用白条分隔"""
    import matplotlib.pyplot as plt
    import os
    import torch
    import numpy as np
    from torch.nn.functional import interpolate
    
    # 创建输出目录
    os.makedirs("attention_vis", exist_ok=True)
    
    # 获取特定层的注意力权重
    layer_attn = attn_weights[layer_idx]
    print(f"第 {layer_idx} 层注意力形状: {layer_attn.shape}")
    print(f"图像形状: {image.shape}")
    
    # 检测相机数量
    if len(image.shape) == 5:  # [batch, cameras, channels, height, width]
        num_cameras = image.shape[1]
        height, width = image.shape[3], image.shape[4]
        
        # 创建横向拼接的图像
        img_list = []
        for cam_idx in range(num_cameras):
            img = image[0, cam_idx].permute(1, 2, 0).cpu() * 255.0
            img_list.append(img.numpy().astype(np.uint8))
        
        # 横向拼接图像
        original_img = np.concatenate(img_list, axis=1)
    elif len(image.shape) == 4:  # [batch, channels, height, width]
        num_cameras = 1
        height, width = image.shape[2], image.shape[3]
        original_img = image[0].permute(1, 2, 0).cpu() * 255.0
        original_img = original_img.numpy().astype(np.uint8)
    else:
        print(f"不支持的图像形状: {image.shape}")
        return
    
    # 确定特征图尺寸 - 每个相机15x20
    feat_h = 15
    feat_w = 20 * num_cameras
    total_feats = feat_h * feat_w
    
    # 计算跳过前2个token后的数据长度
    attn_data_len = layer_attn.shape[2] - 2
    
    # 创建一个列表存储所有图像（原始图像和热图）
    all_images = [original_img]
    all_titles = ["original_img"]
    
    # 白色分隔条高度
    separator_height = 5
    
    # 为每个查询生成热图
    for i in range(num_queries):
        query_idx = i
        
        # 获取该查询的注意力权重 (跳过前两个非图像token)
        query_attn = layer_attn[0, query_idx, 2:]
        
        try:
            # 检查是否有足够的特征点
            if attn_data_len >= total_feats:
                # 重塑为2D特征图
                attention_map = query_attn[:total_feats].reshape(feat_h, feat_w)
                
                # 打印注意力权重的统计信息，便于调试
                attn_min = attention_map.min().item()
                attn_max = attention_map.max().item()
                attn_mean = attention_map.mean().item()
                print(f"查询 {query_idx} 注意力范围: 最小={attn_min:.6f}, 最大={attn_max:.6f}, 平均={attn_mean:.6f}")
                
                # 归一化注意力权重到 [0, 1] 范围
                if attn_max > attn_min:
                    attention_map = (attention_map - attn_min) / (attn_max - attn_min)
                else:
                    attention_map = torch.zeros_like(attention_map)
                
                # 使用最近邻插值上采样到原始图像尺寸
                attention_map = attention_map.unsqueeze(0).unsqueeze(0)
                attention_map = interpolate(attention_map, 
                                           size=(height, width * num_cameras), 
                                           mode='nearest')  # 使用最近邻插值
                attention_map = attention_map[0, 0].cpu().detach().numpy()
                
                # 创建彩色热图（使用hot颜色映射，确保可见度）
                cmap = plt.cm.get_cmap('hot')
                colored_heatmap = cmap(attention_map)
                colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
                
                # 检查热图是否全黑
                if np.mean(colored_heatmap) < 10:  # 如果热图平均值太低（接近黑色）
                    print(f"警告: 查询 {query_idx} 的热图几乎全黑，强制调整对比度")
                    # 强制提高亮度和对比度
                    colored_heatmap = np.clip(colored_heatmap * 5, 0, 255).astype(np.uint8)
                
                # 添加到图像列表
                all_images.append(colored_heatmap)
                all_titles.append(f"step {query_idx}")
                
            else:
                # 如果特征点不够，创建一个最佳匹配的布局
                print(f"警告: 可用特征点({attn_data_len})少于预期({total_feats})")
                
                # # 尝试保持高宽比
                # new_feat_h = min(feat_h, int(np.sqrt(attn_data_len/num_cameras)))
                # new_feat_w = min(int(attn_data_len/new_feat_h), feat_w)
                
                # if new_feat_h * new_feat_w <= attn_data_len:
                #     attention_map = query_attn[:new_feat_h * new_feat_w].reshape(new_feat_h, new_feat_w)
                #     attention_map = attention_map.unsqueeze(0).unsqueeze(0)
                #     attention_map = interpolate(attention_map, 
                #                                size=(height, width * num_cameras), 
                #                                mode='nearest')
                #     attention_map = attention_map[0, 0].cpu().detach().numpy()
                    
                #     # 创建彩色热图
                #     cmap = plt.cm.get_cmap('hot')
                #     colored_heatmap = cmap(attention_map)
                #     colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
                    
                #     # 添加到图像列表
                #     all_images.append(colored_heatmap)
                #     all_titles.append(f"step {query_idx} (调整大小)")
                # else:
                #     # 创建一个空白图像来表示错误
                #     blank = np.ones((height, width * num_cameras, 3), dtype=np.uint8) * 200
                #     cv_text(blank, f"查询 {query_idx} - 数据不足", (10, height//2))
                #     all_images.append(blank)
                #     all_titles.append(f"查询 {query_idx} (数据不足)")
                
        except Exception as e:
            print(f"处理查询 {query_idx} 时出错: {e}")
            # 创建一个红色图像表示错误
            error_img = np.ones((height, width * num_cameras, 3), dtype=np.uint8) * 200
            error_img[:, :, 0] = 240  # 偏红色
            cv_text(error_img, f"查询 {query_idx} 处理失败: {str(e)}", (10, height//2))
            all_images.append(error_img)
            all_titles.append(f"查询 {query_idx} (错误)")
    
    # 创建白色分隔条
    separator = np.ones((separator_height, width * num_cameras, 3), dtype=np.uint8) * 255
    
    # 计算最终图像的总高度
    total_height = sum([img.shape[0] for img in all_images]) + separator_height * (len(all_images) - 1)
    
    # 创建最终的大图像
    final_image = np.ones((total_height, width * num_cameras, 3), dtype=np.uint8) * 128  # 使用灰色背景以便调试
    
    # 填充图像
    y_offset = 0
    for i, img in enumerate(all_images):
        h = img.shape[0]
        final_image[y_offset:y_offset+h, :, :] = img
        y_offset += h
        
        # 如果不是最后一张图像，添加分隔条
        if i < len(all_images) - 1:
            final_image[y_offset:y_offset+separator_height, :, :] = separator
            y_offset += separator_height
    
    # 添加文字标签
    y_offset = 0
    for i, title in enumerate(all_titles):
        y_pos = y_offset + 20  # 距离每个图像顶部20像素
        cv_text(final_image, title, (10, y_pos))
        y_offset += all_images[i].shape[0]
        if i < len(all_images) - 1:
            y_offset += separator_height
    
    import cv2
    cv2.imwrite("attention_vis/multiple_queries_attention_cv2.png", cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print("已使用CV2保存多查询注意力可视化: attention_vis/multiple_queries_attention_cv2.png")

    
    # 保存图像
    # plt.figure(figsize=(12, total_height/80))  # 适当的尺寸
    # plt.imshow(final_image)
    # plt.axis('off')
    # plt.tight_layout(pad=0)
    # plt.savefig("attention_vis/multiple_queries_attention.png", dpi=150, bbox_inches='tight')
    # plt.close()
    
    print("已保存多查询注意力可视化: attention_vis/multiple_queries_attention.png")

def cv_text(img, text, position, font_scale=0.7, color=(0, 0, 0), thickness=2):
    """在图像上添加文字"""
    try:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, position, font, font_scale, color, thickness)
    except ImportError:
        # 如果没有cv2，使用PIL
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            # 使用默认字体
            draw.text(position, text, fill=color)
            img[:] = np.array(pil_img)
        except:
            pass  # 如果PIL也不可用，则跳过文字绘制