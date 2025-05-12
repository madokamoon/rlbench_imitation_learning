import os
import cv2
import numpy as np

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

def main():
    # 指定图片路径
    # 假设图片在项目目录下，如果在其他位置，请相应调整路径
    image_dir = "/home/madoka/python/rlbench_imitation_learning"
    img1_path = os.path.join(image_dir, "3.png")
    img2_path = os.path.join(image_dir, "3.png")
    
    # 检查文件是否存在
    if not os.path.exists(img1_path):
        print(f"错误：文件 {img1_path} 不存在")
        return
    if not os.path.exists(img2_path):
        print(f"错误：文件 {img2_path} 不存在")
        return
    
    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 转换颜色通道从BGR到RGB
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 计算每张图片的熵
    entropy1 = rgb_entropy(img1)
    entropy2 = rgb_entropy(img2)
    
    # 计算互信息
    mi = rgb_mutual_information(img1, img2)
    
    # 输出结果
    print(f"0.png 的熵：{entropy1:.6f}")
    print(f"1.png 的熵：{entropy2:.6f}")
    print(f"0.png 和 1.png 之间的互信息：{mi:.6f}")

if __name__ == "__main__":
    main()