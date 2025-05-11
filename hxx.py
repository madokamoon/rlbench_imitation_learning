import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def natural_sort_key(s):
    """用于自然排序的键函数，提取文件名中的数字"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def calculate_mutual_information_sequence(image_folder):
    """计算图像序列中相邻帧的互信息"""
    # 获取目录中的所有PNG图片
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # 按文件名中的数字排序
    image_files.sort(key=natural_sort_key)
    
    if len(image_files) < 2:
        print("文件夹中图片数量不足，无法计算互信息")
        return None
    
    # 存储互信息结果
    mi_values = []
    frame_pairs = []
    
    # 读取第一张图片
    prev_img_path = os.path.join(image_folder, image_files[0])
    prev_img = cv2.imread(prev_img_path)
    
    # 转换为RGB (OpenCV默认加载为BGR)
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    
    print(f"总共找到 {len(image_files)} 张图片")
    print(f"计算相邻帧的互信息...")
    
    # 遍历后续图片并计算互信息
    for i in tqdm(range(1, len(image_files))):
        # 读取当前图片
        curr_img_path = os.path.join(image_folder, image_files[i])
        curr_img = cv2.imread(curr_img_path)
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        
        # 计算互信息
        mi = rgb_mutual_information(prev_img, curr_img)
        
        # 存储结果
        mi_values.append(mi)
        frame_pairs.append(f"{image_files[i-1]} - {image_files[i]}")
        
        # 当前图片变为下一轮的前一张
        prev_img = curr_img
    
    return mi_values, frame_pairs

def main():
    # 指定根目录
    root_folder = "/home/madoka/python/rlbench_imitation_learning/data/pick_and_lift/50demos/12"
    
    # 获取当前脚本所在目录，用于保存图表
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
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
        
        # 计算互信息序列
        mi_values, frame_pairs = calculate_mutual_information_sequence(image_folder)
        
        if mi_values:
            # 绘制结果图表
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(mi_values) + 1), mi_values, 'b-o', markersize=4)
            plt.grid(True)
            plt.title(f'相邻帧互信息变化 - {camera_dir}')
            plt.xlabel('帧对序号')
            plt.ylabel('互信息值')
            plt.tight_layout()
            
            # 保存图表到脚本所在目录
            output_path = os.path.join(script_dir, f"mutual_information_{camera_dir}.png")
            plt.savefig(output_path)
            print(f"图表已保存至: {output_path}")
            
            # 计算统计信息
            avg_mi = np.mean(mi_values)
            std_mi = np.std(mi_values)
            min_mi = np.min(mi_values)
            max_mi = np.max(mi_values)
            
            print(f"统计信息:")
            print(f"平均互信息: {avg_mi:.6f}")
            print(f"标准差: {std_mi:.6f}")
            print(f"最小值: {min_mi:.6f}")
            print(f"最大值: {max_mi:.6f}")
            
            plt.close()  # 关闭图表以释放内存

if __name__ == "__main__":
    main()