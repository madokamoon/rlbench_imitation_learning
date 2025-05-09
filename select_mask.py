import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取掩码图像为灰度模式
mask_path = '/home/madoka/python/rlbench_imitation_learning/data/pick_and_lift/1demos/0/front_camera_mask/100.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print(f"掩码形状: {mask.shape}")

# 检查图像是否成功读取
if mask is None:
    print(f"无法读取图像: {mask_path}")
    exit()

# 找出唯一灰度值
unique_values = np.unique(mask)
num_values = len(unique_values)

print(f"图像中共有 {num_values} 种不同的灰度值")

# 创建一个彩色图像用于可视化
colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
display_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

# 为每种灰度值分配一个彩色值并标注
cmap = plt.cm.get_cmap('tab20', num_values)
new_colors = (cmap(np.arange(num_values))[:, :3] * 255).astype(np.uint8)

# 为每个区域应用不同的颜色
for i, value in enumerate(unique_values):
    # 创建二值掩码
    binary_mask = (mask == value).astype(np.uint8)
    
    # 使用OpenCV的连通组件分析代替skimage
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    
    # 获取区域颜色
    color = new_colors[i]
    color_bgr = (int(color[2]), int(color[1]), int(color[0]))  # RGB转BGR
    
    # 为彩色可视化图应用颜色
    colored_mask[binary_mask == 1] = color
    
    # 找到面积最大的连通区域用于标注(跳过背景标签0)
    if num_labels > 1:
        # 按面积排序区域(跳过背景)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1  # +1因为跳过了背景
        
        # 获取中心点
        x = int(centroids[largest_label][0])
        y = int(centroids[largest_label][1])
        
        # 在显示图像上标记灰度值
        cv2.putText(display_img, str(value), (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # 输出信息
        print(f"区域 {i+1}: 灰度值 {value}, 面积最大连通块中心点: ({x}, {y}), 区域颜色: {color}")

# 显示结果
cv2.namedWindow('Mask with Labels', cv2.WINDOW_NORMAL)
cv2.imshow('Mask with Labels', display_img)

cv2.namedWindow('Colored Regions', cv2.WINDOW_NORMAL)
cv2.imshow('Colored Regions', cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))

print("按任意键退出")
cv2.waitKey(0)
cv2.destroyAllWindows()


# ACT模型使用示例
# 方法1: 使用单通道掩码
mask_for_model = mask  # 已经是单通道

# 方法2: 扩展为3通道掩码
mask_3channel = np.stack([mask, mask, mask], axis=2)

# 终点 83        （255，0，0）
# 夹爪 35 31 34  （255，0，0）
# 物体 84        （0，255，0）
# 其余设置为      （0，0，0）