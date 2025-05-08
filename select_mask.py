import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

# 读取掩码图像
mask_path = '/home/madoka/python/rlbench_imitation_learning/data/pick_and_lift/2025-05-08-16-10-53/0/front_camera_mask/0.png'
mask = cv2.imread(mask_path)

# 检查图像是否成功读取
if mask is None:
    print(f"无法读取图像: {mask_path}")
else:
    # 将 BGR 转换为 RGB 
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # 将图像重塑为像素列表
    pixels = mask_rgb.reshape(-1, 3)
    
    # 找出唯一颜色
    unique_colors = np.unique(pixels, axis=0)
    num_colors = len(unique_colors)
    
    print(f"图像中共有 {num_colors} 种不同的颜色")
    
    # 创建一个彩色图像来显示不同的区域
    colored_mask = np.zeros_like(mask_rgb)
    
    # 为每种颜色分配一个新的彩色值
    cmap = plt.cm.get_cmap('tab20', num_colors)
    new_colors = (cmap(np.arange(num_colors))[:, :3] * 255).astype(np.uint8)
    
    # 创建颜色映射
    color_map = {}
    for i, color in enumerate(unique_colors):
        color_map[tuple(color)] = new_colors[i]
    
    # 可选: 打印颜色映射
    print("颜色映射:")
    for i, (orig_color, new_color) in enumerate(color_map.items()):
        print(f"区域 {i+1}: 原始颜色 {orig_color} -> 新颜色 {new_color}")
    
    # 为每个区域应用新颜色
    for i in range(mask_rgb.shape[0]):
        for j in range(mask_rgb.shape[1]):
            pixel_color = tuple(mask_rgb[i, j])
            colored_mask[i, j] = color_map[pixel_color]
    
    # 显示原始掩码和彩色区域
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"原始掩码图像 ({num_colors} 种颜色)")
    plt.imshow(mask_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("彩色区分的区域")
    plt.imshow(colored_mask)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()