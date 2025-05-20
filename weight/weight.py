import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

def entropy(image, bins=256):
    """计算图像的熵"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算直方图
    hist, _ = np.histogram(image.ravel(), bins=bins, range=[0, 256])
    
    # 归一化，得到概率分布
    prob = hist / np.sum(hist)
    
    # 去除零概率
    prob = prob[prob > 0]
    
    # 计算熵
    return -np.sum(prob * np.log(prob))

def rgb_entropy(img, bins=256):
    """计算RGB图像三个通道的平均熵"""
    assert len(img.shape) == 3 and img.shape[2] == 3
    
    ent_total = 0
    for c in range(3):
        ent_total += entropy(img[:, :, c], bins=bins)
    return ent_total / 3

def mutual_information(image1, image2, bins=256):
    # 转为灰度（若非灰度输入）
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算联合直方图
    joint_hist, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)

    # 归一化联合分布
    joint_prob = joint_hist / np.sum(joint_hist)

    # 边缘分布
    p1 = np.sum(joint_prob, axis=1)
    p2 = np.sum(joint_prob, axis=0)

    # 外积构造 p1 * p2
    p1_p2 = np.outer(p1, p2)

    # 避免除0、log0
    mask = (joint_prob > 0) & (p1_p2 > 0)

    # 互信息计算
    mi = np.sum(joint_prob[mask] * np.log(joint_prob[mask] / p1_p2[mask]))

    return mi

def rgb_mutual_information(img1, img2, bins=256):
    assert img1.shape == img2.shape and img1.shape[2] == 3

    mi_total = 0
    for c in range(3):
        mi_total += mutual_information(img1[:, :, c], img2[:, :, c], bins=bins)
    return mi_total / 3


def calculate_change_weight(img1, img2, bins=256):
    """计算变化权重：1 - 互信息/最小熵"""
    # 计算两图像的熵
    ent1 = rgb_entropy(img1, bins)
    ent2 = rgb_entropy(img2, bins)
    
    # 计算互信息
    mi = rgb_mutual_information(img1, img2, bins)
    
    # 避免除零
    min_entropy = min(ent1, ent2)
    if min_entropy == 0:
        return 1.0  # 如果熵为零（极少见），返回最大变化权重
    
    # 计算变化权重，范围在[0,1]之间

    # 方式一
    # change_weight = 1.0 - (mi / min_entropy)
    # 方式二
    change_weight = 1.0 - (mi / np.sqrt(ent1 * ent2))
    
    # 确保权重在有效范围内
    return max(0.0, min(1.0, change_weight))





def natural_sort_key(s):
    """用于自然排序的键函数，提取文件名中的数字"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def calculate_image_weights(image_folder, alpha=0.5, beta=0.5):
    """计算图像序列的权重"""
    # 获取目录中的所有PNG图片
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # 按文件名中的数字排序
    image_files.sort(key=natural_sort_key)
    
    if len(image_files) < 2:
        print("文件夹中图片数量不足，无法计算权重")
        return None
    
    # 存储计算结果
    base_weights = []      # 基础权重（熵）
    change_weights = []    # 变化权重
    total_weights = []     # 总权重
    frame_pairs = []       # 帧对
    
    # 读取第一张图片并计算其基础权重
    prev_img_path = os.path.join(image_folder, image_files[0])
    prev_img = cv2.imread(prev_img_path)
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    first_entropy = rgb_entropy(prev_img)
    base_weights.append(first_entropy)
    
    # 第一帧的变化权重设为0（因为没有前一帧）
    change_weights.append(0.0)
    total_weights.append(alpha * first_entropy)
    
    print(f"总共找到 {len(image_files)} 张图片")
    print(f"计算帧权重...")
    
    # 遍历后续图片并计算权重
    for i in tqdm(range(1, len(image_files))):
        # 读取当前图片
        curr_img_path = os.path.join(image_folder, image_files[i])
        curr_img = cv2.imread(curr_img_path)
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        
        # 计算基础权重（熵）
        curr_entropy = rgb_entropy(curr_img)
        base_weights.append(curr_entropy)
        
        # 计算变化权重
        change_weight = calculate_change_weight(prev_img, curr_img)
        change_weights.append(change_weight)
        
        # 计算总权重
        total_weight = alpha * curr_entropy + beta * change_weight
        total_weights.append(total_weight)
        
        # 存储帧对信息
        frame_pairs.append(f"{image_files[i-1]} - {image_files[i]}")
        
        # 当前图片变为下一轮的前一张
        prev_img = curr_img
    
    return {
        "base_weights": base_weights,
        "change_weights": change_weights,
        "total_weights": total_weights,
        "frame_pairs": frame_pairs
    }

def main():
    # 指定根目录
    root_folder = "/home/madoka/python/rlbench_imitation_learning/data/pick_and_lift/30static/0"
    
    # 获取当前脚本所在目录，用于保存图表
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 权重参数
    alpha = 0.5  # 基础权重系数
    beta = 0.5   # 变化权重系数
    
    # 获取根目录下的所有子目录
    camera_dirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    
    if not camera_dirs:
        print(f"在 {root_folder} 下未找到子目录")
        return
    
    print(f"找到 {len(camera_dirs)} 个子目录，开始处理...")
    
    for camera_dir in camera_dirs:
        image_folder = os.path.join(root_folder, camera_dir)
        print(f"\n处理目录: {image_folder}")
        
        # 检查目录中是否有PNG图片
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        if not image_files:
            print(f"目录 {camera_dir} 中未找到PNG图片，跳过")
            continue
        
        # 计算权重
        results = calculate_image_weights(image_folder, alpha, beta)
        
        if results:
            # 准备绘图数据
            base_weights = results["base_weights"]
            change_weights = results["change_weights"]
            total_weights = results["total_weights"]
            
            # 绘制权重变化图表
            plt.figure(figsize=(12, 8))
            
            # 创建帧索引（第一帧索引为0）
            frames = list(range(len(base_weights)))
            
            # 绘制三种权重
            plt.plot(frames, base_weights, 'b-', label=f'基础权重 (熵) × {alpha}')
            plt.plot(frames, change_weights, 'r-', label=f'变化权重 × {beta}')
            plt.plot(frames, total_weights, 'g-', label='总权重')
            
            plt.grid(True)
            plt.title(f'图像权重分析 - {camera_dir}')
            plt.xlabel('帧索引')
            plt.ylabel('权重值')
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(script_dir, f"image_weights_{camera_dir}.png")
            plt.savefig(output_path)
            print(f"图表已保存至: {output_path}")
            
            # 计算统计信息
            print(f"\n统计信息:")
            print(f"基础权重 - 平均值: {np.mean(base_weights):.6f}, 标准差: {np.std(base_weights):.6f}")
            print(f"变化权重 - 平均值: {np.mean(change_weights):.6f}, 标准差: {np.std(change_weights):.6f}")
            print(f"总权重 - 平均值: {np.mean(total_weights):.6f}, 标准差: {np.std(total_weights):.6f}")
            
            plt.close()  # 关闭图表以释放内存

if __name__ == "__main__":
    main()