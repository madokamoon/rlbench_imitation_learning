import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import imageio
from hxxcopy import calculate_image_weights  # 导入原有的权重计算函数

def calculate_multi_view_weights(root_folder, camera_dirs=['front_camera', 'wrist_camera', 'overhead_camera'], 
                                alpha=0.5, beta=0.5):
    """计算多个视角的权重并返回结果"""
    
    all_weights = {}
    frame_count = float('inf')  # 用于存储最小帧数
    
    # 计算每个视角的权重
    for camera_dir in camera_dirs:
        image_folder = os.path.join(root_folder, camera_dir)
        print(f"\n处理视角: {camera_dir}")
        
        if not os.path.exists(image_folder):
            print(f"目录 {image_folder} 不存在，跳过")
            return None
            
        results = calculate_image_weights(image_folder, alpha, beta)
        
        if not results:
            print(f"视角 {camera_dir} 计算权重失败，跳过")
            return None
            
        all_weights[camera_dir] = results
        # 更新最小帧数
        frame_count = min(frame_count, len(results["total_weights"]))
    
    # 确保所有视角有相同的帧数
    print(f"所有视角使用相同的帧数: {frame_count}")
    
    # 准备权重占比数据
    weight_percentages = {camera: [] for camera in camera_dirs}
    
    # 计算每一帧中各视角的权重占比 - 使用变化权重而非总权重
    for frame_idx in range(frame_count):
        # 计算该帧所有视角的变化权重总和
        frame_sum = sum(all_weights[camera]["change_weights"][frame_idx] for camera in camera_dirs)
        
        # 计算每个视角的权重占比
        if frame_sum > 0:  # 避免除以零
            for camera in camera_dirs:
                percentage = (all_weights[camera]["change_weights"][frame_idx] / frame_sum) * 100
                weight_percentages[camera].append(percentage)
        else:
            # 如果总权重为零，则平均分配
            equal_share = 100.0 / len(camera_dirs)
            for camera in camera_dirs:
                weight_percentages[camera].append(equal_share)
    
    return {
        "all_weights": all_weights,
        "weight_percentages": weight_percentages,
        "frame_count": frame_count,
        "camera_dirs": camera_dirs
    }

def create_percentage_plot(data, frame_idx, camera_dirs, labels=None, colors=None):
    """创建单帧的百分比堆叠柱状图"""
    if labels is None:
        labels = camera_dirs
    
    if colors is None:
        colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    percentages = [data["weight_percentages"][camera][frame_idx] for camera in camera_dirs]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制堆叠柱状图
    bottom = 0
    for i, p in enumerate(percentages):
        plt.bar(["Weight Distribution"], [p], bottom=bottom, color=colors[i], label=labels[i])
        bottom += p
    
    # 添加标签和百分比文字
    plt.title(f"Frame {frame_idx}: Camera View Weight Distribution")
    plt.ylabel("Weight Percentage (%)")
    plt.ylim(0, 100)
    
    # 在柱状图上添加百分比标签
    y_offset = 0
    for i, p in enumerate(percentages):
        plt.text(0, y_offset + p/2, f"{p:.1f}%", ha='center', va='center', fontweight='bold')
        y_offset += p
    
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf()

def create_line_plot(data, frame_idx, camera_dirs, labels=None, colors=None):
    """创建单帧的权重百分比折线图"""
    if labels is None:
        labels = [f"{dir}" for dir in camera_dirs]
        
    if colors is None:
        colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    # 准备数据
    frames = list(range(data["frame_count"]))
    
    plt.figure(figsize=(12, 7))
    
    # 绘制每个视角的权重百分比折线图
    for i, camera in enumerate(camera_dirs):
        percentages = data["weight_percentages"][camera]
        plt.plot(frames, percentages, color=colors[i], label=labels[i], linewidth=2)
    
    # 添加当前帧的垂直线
    plt.axvline(x=frame_idx, color='black', linestyle='--', alpha=0.7)
    
    # 在当前帧位置添加每个视角的百分比标签
    for i, camera in enumerate(camera_dirs):
        percentage = data["weight_percentages"][camera][frame_idx]
        plt.text(frame_idx + 1, percentage, f"{percentage:.1f}%", color=colors[i], fontweight='bold')
    
    plt.title(f"Camera View Weight Distribution Over Time (Frame {frame_idx}/{data['frame_count']-1})")
    plt.xlabel("Frame Index")
    plt.ylabel("Weight Percentage (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def create_weight_distribution_gif(data, output_path, plot_type='line', fps=5):
    """创建权重分布的GIF动画"""
    camera_dirs = data["camera_dirs"]
    
    # 自定义标签和颜色
    labels = [f"{dir}" for dir in camera_dirs]
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    frames = []
    
    print("生成GIF帧...")
    for frame_idx in tqdm(range(data["frame_count"])):
        # 根据选择的图表类型创建图表
        if (plot_type == 'bar'):
            fig = create_percentage_plot(data, frame_idx, camera_dirs, labels, colors)
        else:
            fig = create_line_plot(data, frame_idx, camera_dirs, labels, colors)
        
        # 将图表转换为图像
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(image)
        plt.close(fig)
    
    print(f"创建GIF动画...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"GIF已保存至: {output_path}")

def plot_mutual_information_data(data):
    """绘制三个视角的互信息原始数据"""
    plt.figure(figsize=(14, 8))
    frames = list(range(data["frame_count"]))
    camera_dirs = data["camera_dirs"]
    
    # 设置不同视角的颜色
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    # 为每个视角绘制互信息数据（使用change_weights作为互信息指标）
    for i, camera in enumerate(camera_dirs):
        # 获取该视角的变化权重（基于互信息计算）
        mi_data = data["all_weights"][camera]["change_weights"][:data["frame_count"]]
        plt.plot(frames, mi_data, label=f"{camera} - 互信息变化", 
                 color=colors[i], linewidth=2)
    
    plt.title("各视角互信息变化对比")
    plt.xlabel("帧索引")
    plt.ylabel("互信息变化值 (基于1-MI/熵的计算)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "mutual_information_comparison.png")
    plt.savefig(output_path)
    print(f"互信息对比图已保存至: {output_path}")
    plt.close()

def plot_raw_weights_data(data):
    """绘制三个视角的原始权重数据（未经过百分比计算）"""
    plt.figure(figsize=(14, 8))
    frames = list(range(data["frame_count"]))
    camera_dirs = data["camera_dirs"]
    
    # 设置不同视角的颜色
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # 第一个子图：原始变化权重（基于互信息计算的）
    for i, camera in enumerate(camera_dirs):
        # 获取该视角的原始变化权重
        change_weights = data["all_weights"][camera]["change_weights"][:data["frame_count"]]
        ax1.plot(frames, change_weights, label=f"{camera}", 
                color=colors[i], linewidth=2)
    
    ax1.set_title("各视角原始变化权重 (基于1-MI/熵)")
    ax1.set_ylabel("变化权重值")
    ax1.grid(True)
    ax1.legend()
    
    # 第二个子图：基础权重（熵）
    for i, camera in enumerate(camera_dirs):
        # 获取该视角的原始基础权重（熵）
        base_weights = data["all_weights"][camera]["base_weights"][:data["frame_count"]]
        ax2.plot(frames, base_weights, label=f"{camera}", 
                color=colors[i], linewidth=2)
    
    ax2.set_title("各视角原始基础权重 (熵)")
    ax2.set_xlabel("帧索引")
    ax2.set_ylabel("熵值")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图表
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "raw_weights_comparison.png")
    plt.savefig(output_path)
    print(f"原始权重对比图已保存至: {output_path}")
    plt.close()

def main():
    # 指定根目录
    root_folder = "/home/madoka/python/rlbench_imitation_learning/data/pick_and_lift/30static/0"
    
    # 获取根目录下的所有子目录
    available_dirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    
    if not available_dirs:
        print(f"根目录 {root_folder} 中没有找到子目录")
        return
    
    print(f"找到以下子目录: {available_dirs}")
    
    # 筛选出以mask结尾的视角目录
    mask_dirs = [d for d in available_dirs if d.endswith('mask')]
    if not mask_dirs:
        print(f"警告：未找到以'mask'结尾的目录，将使用所有可用目录")
        camera_dirs = available_dirs
    else:
        camera_dirs = mask_dirs
    print(f"将使用以下目录作为相机视角: {camera_dirs}")
    
    # 权重参数
    alpha = 0.5  # 基础权重系数
    beta = 0.5   # 变化权重系数
    
    # 获取当前脚本所在目录，用于保存输出文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"计算多视角权重...")
    data = calculate_multi_view_weights(root_folder, camera_dirs, alpha, beta)
    
    if data:
        # 创建折线图GIF
        line_gif_path = os.path.join(script_dir, "weight_distribution_line.gif")
        create_weight_distribution_gif(data, line_gif_path, plot_type='line', fps=5)
        
        # 创建柱状图GIF
        bar_gif_path = os.path.join(script_dir, "weight_distribution_bar.gif")
        create_weight_distribution_gif(data, bar_gif_path, plot_type='bar', fps=5)
        
        # 另外生成静态的完整折线图
        plt.figure(figsize=(14, 8))
        frames = list(range(data["frame_count"]))
        
        for i, camera in enumerate(camera_dirs):
            percentages = data["weight_percentages"][camera]
            plt.plot(frames, percentages, label=camera, linewidth=2)
        
        plt.title(f"Camera View Weight Distribution Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Weight Percentage (%)")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存静态图
        static_plot_path = os.path.join(script_dir, "weight_distribution_full.png")
        plt.savefig(static_plot_path)
        print(f"静态图表已保存至: {static_plot_path}")
        plt.close()
        
        # 添加互信息原始数据对比图
        plot_mutual_information_data(data)
        
        # 添加原始权重数据对比图
        plot_raw_weights_data(data)

if __name__ == "__main__":
    main()