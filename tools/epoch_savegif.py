import json
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import pathlib
import matplotlib
import argparse
from PIL import Image
import numpy as np
import time
from collections import deque
from tqdm import tqdm  # 添加tqdm用于显示进度条

root_path = pathlib.Path(__file__).parent.parent

# 添加命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("subfolder", type=str, help="epoch子文件夹名称")
args = parser.parse_args()
rootpath = pathlib.Path(__file__).parent
config_file = rootpath.joinpath('config_visualize_epoch.yaml') 
print("配置文件：",config_file)
# 读取 config.yaml
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# 从配置文件获取需要显示的维度、数据路径和图像
show_data = config.get('show_data')
datapath = config.get('datapath')
show_image = config.get('show_image')
# 从配置文件获取目标GIF时长，默认为3秒
target_seconds = config.get('target_seconds') 
giffps= config.get('giffps') 
gifshow = config.get('gifshow')
# 正确顺序：先定义data_folder，再定义json_file
rootpath = pathlib.Path(__file__).parent.parent
data_folder = pathlib.Path(root_path, datapath, args.subfolder)
print("数据文件：",data_folder)
json_file = pathlib.Path(data_folder, 'state.json')

# 从配置文件读取维度索引，而不是使用硬编码值
show_dims = [
    show_data.get('robot_state'), 
    show_data.get('robot_action'), 
    show_data.get('robot_vel_command') 
]

print("使用的维度索引:")
print(f"robot_state: {show_dims[0]}")
print(f"robot_action: {show_dims[1]}")
print(f"robot_vel_command: {show_dims[2]}")

print("开始加载数据...")
start_time = time.time()

# 读取数据
with open(json_file, 'r') as f:
    data = json.load(f)

# 按编号排序
keys = sorted(data.keys(), key=lambda x: int(x))

# 提取多维数据
def extract(data, key, dims):
    arr = []
    for k in keys:
        if key in data[k] and len(data[k][key]) > 0:
            arr.append([data[k][key][d] for d in dims])
        else:
            arr.append([None for _ in dims])
    return arr

# 使用正确的字段名称
robot_state = extract(data, 'robot_state', show_dims[0])
robot_action = extract(data, 'robot_action', show_dims[1])
robot_vel_command = extract(data, 'robot_vel_command', show_dims[2])


# 提取夹爪信息
grasp_state = []
grasp_action = []
for k in keys:
    if 'grasp_state' in data[k] and len(data[k]['grasp_state']) > 0:
        grasp_state.append(data[k]['grasp_state'][0])
    else:
        grasp_state.append(None)


    if 'grasp_action' in data[k] and len(data[k]['grasp_action']) > 0:
        grasp_action.append(data[k]['grasp_action'][0])
    else:
        grasp_action.append(None)

# 检查图像文件夹是否存在
camera_folders = []
for camera in show_image[:3]:  # 最多取前三个摄像头
    camera_folder = data_folder / camera
    if camera_folder.exists():
        camera_folders.append(camera_folder)
    else:
        print(f"警告：摄像头文件夹 {camera_folder} 不存在")
        camera_folders.append(None)

# 填充到3个摄像头
while len(camera_folders) < 3:
    camera_folders.append(None)

# 修改为4行2列的图像布局
fig, axs = plt.subplots(4, 2, figsize= (20, 15)) 
titles_left = ['robot_state', 'robot_action', 'robot_vel_command', 'grasp_info']
datas = [robot_state, robot_action, robot_vel_command]
lines = []

# 定义颜色列表，足够多即可
colors = list(matplotlib.colormaps['tab10'].colors) + list(matplotlib.colormaps['tab20'].colors)

# 修改绘制左侧前三个子图（机械臂数据）的代码，实现双y轴
# 修改前三个子图的部分（替换原来的代码块）
# 绘制左侧四个子图（数据图表）
lines = []
twin_axs = []  # 存储第二个y轴的引用

for i in range(4):
    axs[i, 0].set_title(titles_left[i])
    axs[i, 0].set_xlim(0, len(keys))
    
    if i < 3:  # 前三个子图（机械臂数据）- 使用双y轴
        # 创建第二个y轴
        twin_ax = axs[i, 0].twinx()
        twin_axs.append(twin_ax)
        
        # 设置左右y轴的标签
        axs[i, 0].set_ylabel('Position (m)', color='blue')
        twin_ax.set_ylabel('Angle (rad)', color='red')
        
        # 分别计算位置和角度数据的最大最小值
        pos_dims = show_dims[i][:3]  # 前三维 [0,1,2] 是位置
        ang_dims = show_dims[i][3:]  # 后三维 [3,4,5] 是角度
        
        # 计算位置的min/max
        pos_flat = []
        for arr in datas[i]:
            for j, d in enumerate(pos_dims):
                if j < len(arr) and arr[j] is not None:
                    pos_flat.append(arr[j])
        pos_minv = min(pos_flat) if pos_flat else -1
        pos_maxv = max(pos_flat) if pos_flat else 1
        axs[i, 0].set_ylim(pos_minv - 0.01, pos_maxv + 0.01)
        
        # 计算角度的min/max
        ang_flat = []
        for arr in datas[i]:
            for j, d in enumerate(ang_dims):
                idx = j + 3  # 角度从索引3开始
                if idx < len(arr) and arr[idx] is not None:
                    ang_flat.append(arr[idx])
        ang_minv = min(ang_flat) if ang_flat else -1
        ang_maxv = max(ang_flat) if ang_flat else 1
        twin_ax.set_ylim(ang_minv - 0.01, ang_maxv + 0.01)
        
        # 为每个维度画一条线，分别使用不同的y轴
        line_list = []
        
        # 绘制位置数据 (左侧y轴)
        for j, d in enumerate(pos_dims):
            color = colors[d % len(colors)]
            line, = axs[i, 0].plot([], [], lw=2, label=f'dim_{d} (m)', color=color)
            line_list.append(line)
        
        # 绘制角度数据 (右侧y轴)
        for j, d in enumerate(ang_dims):
            color = colors[(d+10) % len(colors)]  # 使用不同的颜色区域
            line, = twin_ax.plot([], [], lw=2, label=f'dim_{d} (rad)', color=color)
            line_list.append(line)
        
        # 合并两个轴的图例
        lines1, labels1 = axs[i, 0].get_legend_handles_labels()
        lines2, labels2 = twin_ax.get_legend_handles_labels()
        twin_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        lines.append(line_list)

    else:  # 第四个子图（夹爪信息）- 不变
        # 计算夹爪信息的全局最小最大值
        grasp_flat = [v for v in grasp_state if v is not None] + [v for v in grasp_action if v is not None]
        grasp_minv = min(grasp_flat) if grasp_flat else -1
        grasp_maxv = max(grasp_flat) if grasp_flat else 1
        axs[i, 0].set_ylim(grasp_minv - 0.01, grasp_maxv + 0.01)
        # 添加夹爪状态和动作的线
        line_state, = axs[i, 0].plot([], [], lw=2, label='grasp_state', color=colors[0])
        line_action, = axs[i, 0].plot([], [], lw=2, label='grasp_action', color=colors[1])
        grasp_lines = [line_state, line_action]
        axs[i, 0].legend()
        lines.append(grasp_lines)


# 配置右侧前三个子图（摄像头图像）
for i in range(3):
    if i < len(show_image):
        axs[i, 1].set_title(show_image[i])
    axs[i, 1].axis('off')  # 关闭坐标轴

# 配置右侧第四个子图（信息显示）
axs[3, 1].axis('off')  # 关闭坐标轴
info_text = axs[3, 1].text(0.5, 0.5, '', ha='center', va='center', fontsize=12)

# 图像对象列表（用于更新）
img_objects = [None, None, None]

time_diffs = deque(maxlen=4)
frame_rates = deque(maxlen=4)
last_timestamp = None

# 在全局变量区域添加新的变量（在time_diffs定义之前）
first_timestamp = None
max_frame_rate = 0
min_frame_rate = float('inf')
total_frame_rate = 0
frame_count = 0
max_frame_rate_idx = 0  # 记录最大帧率对应的帧编号
min_frame_rate_idx = 0  # 记录最小帧率对应的帧编号

def update(frame):
    if frame < len(keys):
        key = keys[frame]
        timestamp = data[key].get("timestamp", None)
        
        # 更新左侧前三个图（机械臂数据）
        for i in range(3):
            pos_dims = show_dims[i][:3]  # 前三维 [0,1,2] 是位置
            ang_dims = show_dims[i][3:]  # 后三维 [3,4,5] 是角度
            
            # 更新位置数据 (前三维)
            for j, d in enumerate(pos_dims):
                y = [arr[j] if len(arr) > j and arr[j] is not None else float('nan') for arr in datas[i][:frame]]
                lines[i][j].set_data(range(frame), y)
            
            # 更新角度数据 (后三维)
            for j, d in enumerate(ang_dims):
                idx = j + 3  # 角度从索引3开始
                y = [arr[idx] if len(arr) > idx and arr[idx] is not None else float('nan') for arr in datas[i][:frame]]
                lines[i][j+len(pos_dims)].set_data(range(frame), y)
        
        # 更新左侧第四个图（夹爪信息）- 保持不变
        state_y = [v if v is not None else float('nan') for v in grasp_state[:frame]]
        action_y = [v if v is not None else float('nan') for v in grasp_action[:frame]]
        lines[3][0].set_data(range(frame), state_y)
        lines[3][1].set_data(range(frame), action_y)
        
        
        # 更新右侧前三个图（摄像头图像）
        for i in range(3):
            if camera_folders[i] is not None:
                img_path = camera_folders[i] / f"{key}.png"
                if img_path.exists():
                    # 清除之前的图像
                    # if img_objects[i] is not None:
                    #     img_objects[i].remove()
                    
                    # 读取并显示新图像
                    img = plt.imread(img_path)
                    img_objects[i] = axs[i, 1].imshow(img)
                    axs[i, 1].set_title(f"{show_image[i]}")
        
        # 更新右侧第四个图（信息显示）
        global last_timestamp, first_timestamp, max_frame_rate, min_frame_rate, total_frame_rate, frame_count, max_frame_rate_idx, min_frame_rate_idx
        
        # 初始化第一帧时间戳
        if first_timestamp is None and timestamp is not None:
            first_timestamp = timestamp
        
        if timestamp is not None and last_timestamp is not None:
            time_diff = timestamp - last_timestamp
            frame_rate = 1.0 / time_diff if time_diff > 0 else 0
            
            # 更新统计信息
            if frame_rate > 0:
                if frame_rate > max_frame_rate:
                    max_frame_rate = frame_rate
                    max_frame_rate_idx = frame
                if frame_rate < min_frame_rate:
                    min_frame_rate = frame_rate
                    min_frame_rate_idx = frame
                total_frame_rate += frame_rate
                frame_count += 1
            
            time_diffs.append(round(time_diff, 4))
            frame_rates.append(round(frame_rate, 2))
        
        # 构建显示字符串
        current_time = timestamp - first_timestamp if first_timestamp is not None else 0
        avg_frame_rate = total_frame_rate / frame_count if frame_count > 0 else 0
        
        info_str = f"Frame: {key}\n\nTimestamp: {timestamp}\n"
        info_str += f"Current Time: {current_time:.3f}s\n"
        info_str += f"Max FPS: {max_frame_rate:.2f} (Frame {max_frame_rate_idx})\n"
        info_str += f"Min FPS: {min_frame_rate:.2f} (Frame {min_frame_rate_idx})\n"
        info_str += f"Avg FPS: {avg_frame_rate:.2f} (Total {frame_count})\n\n"
        
        # 使用enumerate获取实际的帧序号，存储在time_diffs和frame_rates中的最近4个帧的统计信息
        frame_history = list(range(max(0, frame-len(time_diffs)), frame))
        for frame_idx, (td, fr) in zip(frame_history, zip(time_diffs, frame_rates)):
            info_str += f"{frame_idx:5d} | {td:7.4f} | {fr:7.2f}\n"
        
        info_text.set_text(info_str)
        last_timestamp = timestamp
        
    return [l for sub in lines for l in sub] + [obj for obj in img_objects if obj is not None] + [info_text]



# 根据 gifshow 参数决定是显示还是保存
if gifshow:
    # 只显示，不保存 - 使用全部帧以获得流畅效果
    print("仅显示动态可视化过程，不保存GIF...")
    ani = FuncAnimation(fig, update, frames=len(keys)+1, interval=100, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()
else:
    # 根据目标时长自动计算采样步长
    target_frames = target_seconds * giffps  # 总目标帧数
    gif_save_step = max(1, round(len(keys) / target_frames))  # 至少为1
    print(f"总帧数: {len(keys)}, gif帧率: {giffps}, gif长度: {target_seconds}s, 采样步长: {gif_save_step}")

    # 只保存，不显示 - 使用采样帧以加快保存
    print("仅保存GIF，不显示动画...")
    frame_indices = list(range(0, len(keys), gif_save_step))
    ani = FuncAnimation(fig, update, frames=frame_indices, interval=100, blit=True, repeat=False)
    plt.tight_layout()
    
    # 保存为GIF文件，在数据文件夹下
    print(f"正在将动画保存为GIF文件...")
    output_filename = pathlib.Path(data_folder, "visualization.gif")
    writer = PillowWriter(fps=giffps)
    
    # 使用tqdm显示进度条
    with tqdm(total=len(frame_indices), desc="保存进度") as pbar:
        save_start_time = time.time()
        
        def progress_callback(current_frame, total_frames):
            pbar.update(1)
        
        # 保存动画
        ani.save(str(output_filename), writer=writer, 
                 progress_callback=progress_callback if hasattr(PillowWriter, 'saving') else None)
        
        # 如果PillowWriter不支持progress_callback，则手动更新进度条
        if not hasattr(PillowWriter, 'saving'):
            # 完成最终更新
            pbar.update(len(frame_indices))

    save_duration = time.time() - save_start_time
    print(f"动画已保存为: {output_filename}")
    print(f"保存用时: {save_duration:.2f}秒")

# # 如果directshow=True，则等待用户关闭窗口
# if gifshow:
#     print("请关闭图形窗口继续...")
#     plt.ioff()  # 关闭交互模式
#     plt.show()  # 阻塞等待窗口关闭