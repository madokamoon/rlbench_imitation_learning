# 一些参数说明：
# image.shape： [batch, cameras, channels, height, width]
# flow_data.shape： [batch, cameras, channels, height, width]
# attn_weights： [解码器层数, batch_size, chunk_size, 902]  # 对于3个相机就是902 不固定
# layer_attn = attn_weights[-1]  最后一层shape：[batch_size, chunk_size, 902] # 对于3个相机就是902 不固定
# attn_data_len = layer_attn.shape[2] - 2  计算跳过前2个token后的数据长度
# 确定特征图尺寸 - 每个相机15x20
# feat_h = 15
# feat_w = 20 * num_cameras
# total_feats = feat_h * feat_w
# query_attn = layer_attn[batch_size, chunk_size, 2:]   跳过前两个非图像token的注意力，对于3个相机就是900 不固定
# attention_map = query_attn[:total_feats].reshape(feat_h, feat_w)



def visualize_multiple_attentions(image, attn_weights, num_queries=15, layer_idx=-1, avg_attention_map=None):
    """将原始图像和多个查询的注意力热图竖向拼接为一张长条图像"""
    import matplotlib.pyplot as plt
    import os, cv2
    import numpy as np
    from torch.nn.functional import interpolate
    
    os.makedirs("attention_vis", exist_ok=True)
    
    # 获取注意力权重和处理批次
    layer_attn = attn_weights[layer_idx]
    print(f"注意力形状: {layer_attn.shape}, 图像形状: {image.shape}")
    layer_attn_batch0 = layer_attn[0:1]  # 只取第一个批次
    
    # 处理图像和确定相机数量
    if len(image.shape) == 5:  # [batch, cameras, channels, height, width]
        num_cameras = image.shape[1]
        height, width = image.shape[3], image.shape[4]
        img_list = [image[0, cam_idx].permute(1, 2, 0).cpu().detach().numpy() * 255.0 for cam_idx in range(num_cameras)]
        original_img = np.concatenate([img.astype(np.uint8) for img in img_list], axis=1)
    elif len(image.shape) == 4:  # [batch, channels, height, width]
        num_cameras = 1
        height, width = image.shape[2], image.shape[3]
        original_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    else:
        print(f"不支持的图像形状: {image.shape}")
        return
    
    # 设置特征图尺寸
    feat_h, feat_w = 15, 20 * num_cameras
    total_feats = feat_h * feat_w
    attn_data_len = layer_attn_batch0.shape[2] - 2
    
    # 初始化图像列表
    all_images = [original_img]
    all_titles = ["原始图像"]
    
    # 处理热图的通用函数
    def process_attention_map(attn_tensor, title):
        try:
            # 重塑并归一化 - 先分离梯度
            attn_tensor = attn_tensor.detach()
            attn_2d = attn_tensor.reshape(feat_h, feat_w)
            attn_min, attn_max = attn_2d.min().item(), attn_2d.max().item()
            
            if attn_max > attn_min:
                attn_2d = (attn_2d - attn_min) / (attn_max - attn_min)
            else:
                attn_2d = torch.zeros_like(attn_2d)
            
            # 上采样并转换为热图
            attn_2d = interpolate(attn_2d.unsqueeze(0).unsqueeze(0), 
                                size=(height, width * num_cameras), 
                                mode='nearest')[0, 0].cpu().numpy()
            
            colored_map = (plt.cm.get_cmap('hot')(attn_2d)[:, :, :3] * 255).astype(np.uint8)
            
            # 检查并调整对比度
            if np.mean(colored_map) < 10:
                colored_map = np.clip(colored_map * 5, 0, 255).astype(np.uint8)
                
            all_images.append(colored_map)
            all_titles.append(title)
            
        except Exception as e:
            print(f"处理{title}时出错: {e}")
            error_img = np.ones((height, width * num_cameras, 3), dtype=np.uint8) * 200
            error_img[:, :, 0] = 240  # 偏红色
            cv_text(error_img, f"{title}处理失败: {str(e)}", (10, height//2))
            all_images.append(error_img)
            all_titles.append(f"{title}(错误)")
    
    # 处理平均注意力图
    if avg_attention_map is not None:
        process_attention_map(avg_attention_map[0, 0], "平均注意力图")
    
    # 处理单个查询的注意力图
    for i in range(min(num_queries, layer_attn_batch0.shape[1])):
        if attn_data_len >= total_feats:
            process_attention_map(layer_attn_batch0[0, i, 2:2+total_feats], f"step {i}")
        else:
            print(f"警告: 可用特征点({attn_data_len})少于预期({total_feats})")
    
    # 创建最终图像
    separator = np.ones((5, width * num_cameras, 3), dtype=np.uint8) * 255
    final_image = create_stacked_image(all_images, separator, all_titles)
    
    return final_image



def create_stacked_image(images, separator, titles):
    """创建竖向堆叠的图像"""
    import numpy as np
    
    total_height = sum([img.shape[0] for img in images]) + separator.shape[0] * (len(images) - 1)
    final_image = np.ones((total_height, images[0].shape[1], 3), dtype=np.uint8) * 128
    
    y_offset = 0
    for i, img in enumerate(images):
        h = img.shape[0]
        final_image[y_offset:y_offset+h, :, :] = img
        cv_text(final_image, titles[i], (10, y_offset + 20))
        y_offset += h
        
        if i < len(images) - 1:
            final_image[y_offset:y_offset+separator.shape[0], :, :] = separator
            y_offset += separator.shape[0]
            
    return final_image

def cv_text(img, text, position, font_scale=0.7, color=(0, 0, 0), thickness=2):
    """在图像上添加文字"""
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, position, font, font_scale, color, thickness)
