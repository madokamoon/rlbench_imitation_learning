import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from sklearn.decomposition import PCA
import os

def apply_sam_and_pca_visualization():
    # 1. 加载SAM模型
    print("加载SAM模型...")
    sam_checkpoint = "/home/madoka/python/sam_vit_h_4b8939.pth"  
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    
    # 2. 读取图像
    print("读取图像...")
    image_path = "tools/0.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像文件 '{image_path}'")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. 获取SAM嵌入向量
    print("提取SAM嵌入向量...")
    predictor.set_image(image_rgb)
    
    # 获取图像嵌入
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    
    # 4. 处理嵌入向量以便于可视化
    # 获取特征维度
    print("处理嵌入向量...")
    h, w = image.shape[:2]
    
    # 从模型中获取嵌入尺寸
    embedding_h = image_embedding.shape[2]  # 通常是图像高度的1/16
    embedding_w = image_embedding.shape[3]  # 通常是图像宽度的1/16
    
    # 将嵌入向量重塑为 (C, H*W) 形式
    features = image_embedding.squeeze().reshape(image_embedding.shape[1], -1).T
    print(f"嵌入向量形状: {features.shape}")
    
    # 5. 应用PCA降维
    print("应用PCA降维...")
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features)
    
    # 将PCA结果重塑为原始嵌入尺寸
    pca_visualization = features_pca.reshape(embedding_h, embedding_w, 3)
    
    # 归一化到0-1范围以便于可视化
    pca_visualization = (pca_visualization - pca_visualization.min()) / (pca_visualization.max() - pca_visualization.min())
    
    # 6. 放大PCA可视化结果以匹配原图尺寸
    pca_visualization_resized = cv2.resize(pca_visualization, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 7. 可视化并保存结果
    # 创建输出目录
    output_dir = "pca_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 原始图像
    fig = plt.figure(figsize=(10, 10))  # 使用正方形尺寸
    plt.imshow(image_rgb, aspect='equal')  # 保持图像原始比例
    plt.title("原始图像")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/original_image.png", dpi=300, bbox_inches='tight', pad_inches=0)  # 消除额外边距
    
    # PCA可视化 (使用新窗口)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(pca_visualization_resized, aspect='equal')
    plt.title("SAM嵌入向量的PCA可视化")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_visualization.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
    # 可视化PCA的3个主成分
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        component = features_pca[:, i].reshape(embedding_h, embedding_w)
        component_normalized = (component - component.min()) / (component.max() - component.min())
        component_resized = cv2.resize(component_normalized, (w, h), interpolation=cv2.INTER_LINEAR)
        
        axes[i].imshow(component_resized, cmap='viridis', aspect='equal')  # 保持比例
        axes[i].set_title(f"PCA 主成分 {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_components.png", dpi=300, bbox_inches='tight')
    
    # 计算解释方差
    explained_variance = pca.explained_variance_ratio_
    
    # 可视化解释方差
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, 4), explained_variance)
    plt.xlabel('主成分')
    plt.ylabel('解释方差比')
    plt.title('PCA主成分的解释方差')
    plt.xticks(range(1, 4))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/explained_variance.png", dpi=300)
    
    print(f"所有图像已保存到 {output_dir} 目录")
    print(f"主成分解释方差比: {explained_variance}")
    
    # 显示所有图形（不同窗口）
    plt.show()

if __name__ == "__main__":
    apply_sam_and_pca_visualization()