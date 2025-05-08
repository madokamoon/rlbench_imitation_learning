import json
import yaml
import matplotlib.pyplot as plt
import pathlib
import matplotlib
import argparse
import numpy as np

# 添加命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("subfolder", type=str, help="epoch子文件夹名称")
args = parser.parse_args()

# 读取配置文件
rootpath = pathlib.Path(__file__).parent
config_file = rootpath.joinpath('config_visualize_epoch.yaml')
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# 从配置文件获取数据设置
show_data = config.get('show_data')
datapath = config.get('datapath')

# 设置数据路径
root_path = pathlib.Path(__file__).parent.parent
data_folder = pathlib.Path(root_path, datapath, args.subfolder)
json_file = pathlib.Path(data_folder, 'state.json')

# 从配置读取维度索引
show_dims = [
    show_data.get('robot_state'), 
    show_data.get('robot_action'), 
    show_data.get('robot_vel_command') 
]

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

# 提取各类数据
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

# 创建2x2布局的图表
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()

titles = ['robot_state', 'robot_action', 'robot_vel_command', 'grasp_info']
datas = [robot_state, robot_action, robot_vel_command]
colors = list(matplotlib.colormaps['tab10'].colors) + list(matplotlib.colormaps['tab20'].colors)

# 绘制前三个子图（机械臂数据）
for i in range(3):
    ax = axs[i]
    ax.set_title(titles[i])
    ax.set_xlim(0, len(keys))
    
    # 创建双y轴
    twin_ax = ax.twinx()
    
    # 设置左右y轴的标签
    ax.set_ylabel('Position (m)', color='blue')
    twin_ax.set_ylabel('Angle (rad)', color='red')
    
    # 获取位置和角度数据的索引
    pos_dims = show_dims[i][:3]  # 前三维是位置
    ang_dims = show_dims[i][3:]  # 后三维是角度
    
    # 计算位置和角度的y轴范围
    pos_flat = [arr[j] for arr in datas[i] for j in range(min(len(arr), 3)) if j < len(arr) and arr[j] is not None]
    ang_flat = [arr[j] for arr in datas[i] for j in range(3, min(len(arr), 6)) if j < len(arr) and arr[j] is not None]
    
    pos_minv = min(pos_flat) if pos_flat else -1
    pos_maxv = max(pos_flat) if pos_flat else 1
    ang_minv = min(ang_flat) if ang_flat else -1
    ang_maxv = max(ang_flat) if ang_flat else 1
    
    ax.set_ylim(pos_minv - 0.01, pos_maxv + 0.01)
    twin_ax.set_ylim(ang_minv - 0.01, ang_maxv + 0.01)
    
    # 绘制位置数据
    for j, d in enumerate(pos_dims):
        y = [arr[j] if j < len(arr) and arr[j] is not None else float('nan') for arr in datas[i]]
        ax.plot(range(len(keys)), y, lw=2, label=f'dim_{d} (m)', color=colors[d % len(colors)])
    
    # 绘制角度数据
    for j, d in enumerate(ang_dims):
        idx = j + 3
        y = [arr[idx] if idx < len(arr) and arr[idx] is not None else float('nan') for arr in datas[i]]
        twin_ax.plot(range(len(keys)), y, lw=2, label=f'dim_{d} (rad)', color=colors[(d+10) % len(colors)])
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = twin_ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# 绘制夹爪信息
axs[3].set_title(titles[3])
axs[3].set_xlim(0, len(keys))
grasp_flat = [v for v in grasp_state if v is not None] + [v for v in grasp_action if v is not None]
grasp_minv = min(grasp_flat) if grasp_flat else 0
grasp_maxv = max(grasp_flat) if grasp_flat else 1
axs[3].set_ylim(grasp_minv - 0.01, grasp_maxv + 0.01)

# 绘制夹爪状态和动作
axs[3].plot(range(len(keys)), [v if v is not None else float('nan') for v in grasp_state], 
           lw=2, label='grasp_state', color=colors[0])
axs[3].plot(range(len(keys)), [v if v is not None else float('nan') for v in grasp_action], 
           lw=2, label='grasp_action', color=colors[1])
axs[3].legend()

plt.tight_layout()
plt.savefig(f"{data_folder}/mechanical_arm_data.png", dpi=150)
plt.close()
print(f"图表已保存至 {data_folder}/mechanical_arm_data.png")

# 添加六个关节图表 - 新增功能
joint_names = ['poxX', 'poxY', 'poxZ', 'angleX', 'angleY', 'angleZ']
fig2, axs2 = plt.subplots(2, 3, figsize=(18, 12))
axs2 = axs2.flatten()

# 绘制六个关节的状态和动作对比图
for joint_idx in range(6):
    ax = axs2[joint_idx]
    ax.set_title(f"{joint_idx}: {joint_names[joint_idx]}")
    ax.set_xlim(0, len(keys))
    
    # 提取该关节的状态和动作数据
    state_vals = [arr[joint_idx] if joint_idx < len(arr) and arr[joint_idx] is not None else float('nan') 
                 for arr in robot_state]
    action_vals = [arr[joint_idx] if joint_idx < len(arr) and arr[joint_idx] is not None else float('nan') 
                  for arr in robot_action]
    
    # 计算y轴范围
    all_vals = [v for v in state_vals + action_vals if not np.isnan(v)]
    if all_vals:
        minv = min(all_vals) - 0.01
        maxv = max(all_vals) + 0.01
    else:
        minv, maxv = -1, 1
    
    ax.set_ylim(minv, maxv)
    
    # 设置y轴标签
    unit = "(m)" if joint_idx < 3 else "(rad)"

    ax.set_ylabel(f"{unit}")
    
    # 绘制状态和动作数据
    ax.plot(range(len(keys)), state_vals, lw=2, label='state', color=colors[0])
    ax.plot(range(len(keys)), action_vals, lw=2, label='action', color=colors[1])
    ax.legend(loc='upper right')
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f"{data_folder}/joint_comparison.png", dpi=150)
plt.close()
print(f"关节对比图已保存至 {data_folder}/joint_comparison.png")