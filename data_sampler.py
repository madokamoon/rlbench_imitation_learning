import numpy as np
import os
import h5py
from pprint import pprint
import yaml
import importlib
import json
import pathlib
from PIL import Image
import time, datetime
from tqdm import tqdm 
from pyrep.const import RenderMode

# 导入RLBench相关模块
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import GripperJointPosition
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


class RLBenchProcessor:
    """用于RLBench环境的数据采样与轨迹执行的综合处理器"""
    
    def __init__(self, config_path='data_sampler.yaml'):
        """
        初始化处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
 
        # 从配置中获取参数
        self.save_path_head = self.config.get('save_path_head', 'data')
        self.save_path_end = self.config.get('save_path_end', '')
        self.taskclassname = self.config['taskclassname']
        self.num_demos = self.config.get('num_demos', 1)
        self.image_width = self.config['image']['width']
        self.image_height = self.config['image']['height']
        self.camera_names = self.config['cameras']
        self.robot_setup = self.config['robot_setup']
        
        # 确定运行模式 (采样 or 反应)
        self.reaction_mode = self.config.get('reaction', False)
        if self.reaction_mode:
            self.data_path = self.config['data_path']

        # 初始化环境和任务相关变量
        self.env = None
        self.task = None
        self.obs_config = None

    def _check_and_make(self, directory):
        """创建目录（如果不存在）"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    def _check_path_exists(self, directory):
        """检查路径是否存在"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"路径不存在: {directory}")
        return True

    def _create_observation_config(self):
        """创建观测配置"""
        obs_config = ObservationConfig()

        cameras_dict = {
            'front_camera': obs_config.front_camera,
            'wrist_camera': obs_config.wrist_camera,
            'left_shoulder_camera': obs_config.left_shoulder_camera,
            'right_shoulder_camera': obs_config.right_shoulder_camera,
            'overhead_camera': obs_config.overhead_camera
        }

        # 先关闭所有相机
        for camera in cameras_dict.values():
            camera.rgb = False
            camera.depth = False
            camera.mask = False

        # 只启用配置中指定的相机
        for camera_name in self.camera_names:
            if camera_name in cameras_dict:
                cameras_dict[camera_name].rgb = True
                # 在数据采样模式下启用深度和掩码
                if not self.reaction_mode:
                    cameras_dict[camera_name].depth = True
                    cameras_dict[camera_name].mask = True
                    cameras_dict[camera_name].depth_in_meters = True  # 将深度存储为 0-1 归一化值
                    cameras_dict[camera_name].masks_as_one_channel = True  # 将掩码存储为单通道图像
                cameras_dict[camera_name].image_size = (self.image_width, self.image_height)

        # 设置其他观测参数
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.gripper_open = True
        obs_config.gripper_pose = True 
        obs_config.gripper_joint_positions = True
        obs_config.gripper_matrix = True
        obs_config.gripper_touch_forces = False
        # obs_config.record_gripper_closing = True  # 会卡很久
          
        return obs_config

    def setup_environment(self):
        """设置并启动RLBench环境"""
        # 创建观测配置
        self.obs_config = self._create_observation_config()
        
        # 打印观测配置信息
        print("\n ObservationConfig 属性:")
        pprint(self.obs_config.__dict__)
        print("\n")
        
        print("设置RLBench环境...")
        
        # 根据模式选择不同的控制方式
        if self.reaction_mode:
            # 反应模式：使用末端位姿控制
            action_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),
                gripper_action_mode=GripperJointPosition(absolute_mode=True))
            print("使用末端位姿控制模式进行轨迹执行")
        else:
            # 采样模式：使用关节速度控制
            action_mode = MoveArmThenGripper(
                arm_action_mode=JointVelocity(),
                gripper_action_mode=GripperJointPosition(absolute_mode=True))
            print("使用关节速度控制模式进行数据采样")
            
        # 创建并配置RLBench环境
        self.env = Environment(
            action_mode=action_mode,
            obs_config=self.obs_config, 
            headless=False,
# robot_setup=self.robot_setup
)
            
        self.env.launch()
        print("环境启动成功")

    def load_task(self):
        """加载指定的任务"""
        print(f"加载任务: {self.taskclassname}")
        module = importlib.import_module('rlbench.tasks')
        task_class = getattr(module, self.taskclassname)
        self.task = self.env.get_task(task_class)
        print("任务加载成功")

    def save_demo_raw(self, demo, example_path, ex_idx):
        """
        保存演示数据为原始的PNG和JSON格式
        
        Args:
            demo: 演示数据
            example_path: 保存路径
            ex_idx: 示例索引
        """
        # 创建示例文件夹
        episode_folder = pathlib.Path(example_path).joinpath(f"{ex_idx}")
        episode_folder.mkdir(parents=True, exist_ok=True)
        
        # 创建相机文件夹
        for cam_name in self.camera_names:
            camera_folder = episode_folder.joinpath(cam_name)
            camera_folder_depth = episode_folder.joinpath(cam_name+"_depth")
            camera_folder_mask = episode_folder.joinpath(cam_name+"_mask")
            camera_folder.mkdir(parents=True, exist_ok=True)
            camera_folder_depth.mkdir(parents=True, exist_ok=True)
            camera_folder_mask.mkdir(parents=True, exist_ok=True)

        
        # 初始化数据存储
        state_data = {}
        
        # 使用tqdm为循环添加进度条
        for i, obs in tqdm(enumerate(demo), total=len(demo), desc=f"演示 {ex_idx} 处理进度"):
            # 获取当前时间戳
            current_timestamp = time.time()
            
            # 保存上一帧的动作数据
            if i > 0:
                # 对于GripperJointPosition, 将夹爪开合度转换为关节位置
                prev_frame_data['grasp_action'] = [obs.gripper_open]
                prev_frame_data['robot_joint_action'] = obs.joint_positions.tolist()
                prev_frame_data['robot_action'] = obs.gripper_pose.tolist()
                state_data[str(i-1)] = prev_frame_data

            # 当前帧的状态数据
            frame_data = {
                'timestamp': current_timestamp,
                'grasp_state': [obs.gripper_open],
                'gripper_joint_positions': obs.gripper_joint_positions.tolist() if obs.gripper_joint_positions is not None else None,
                'robot_joint_state': obs.joint_positions.tolist(),
                'robot_joint_vel': obs.joint_velocities.tolist() if obs.joint_velocities is not None else None,
                'robot_state': obs.gripper_pose.tolist(),
                'robot_state_matrix': obs.gripper_matrix.tolist() if obs.gripper_matrix is not None else None,
            }

            prev_frame_data = frame_data

            # 最后一帧也需要保存动作
            if i == len(demo) - 1:
                frame_data['grasp_action'] = [obs.gripper_open]
                frame_data['robot_joint_action'] = obs.joint_positions.tolist()
                frame_data['robot_action'] = obs.gripper_pose.tolist()
                state_data[str(i)] = frame_data

            # 使用字典映射相机属性名到观测对象的属性
            camera_mapping = {
                'front_camera': {
                    'rgb': 'front_rgb',
                    'depth': 'front_depth',
                    'mask': 'front_mask'
                },
                'wrist_camera': {
                    'rgb': 'wrist_rgb',
                    'depth': 'wrist_depth',
                    'mask': 'wrist_mask'
                },
                'left_shoulder_camera': {
                    'rgb': 'left_shoulder_rgb',
                    'depth': 'left_shoulder_depth',
                    'mask': 'left_shoulder_mask'
                },
                'right_shoulder_camera': {
                    'rgb': 'right_shoulder_rgb',
                    'depth': 'right_shoulder_depth',
                    'mask': 'right_shoulder_mask'
                },
                'overhead_camera': {
                    'rgb': 'overhead_rgb',
                    'depth': 'overhead_depth',
                    'mask': 'overhead_mask'
                }
            }

            # 遍历所有相机
            for camera_name in self.camera_names:
                if camera_name not in camera_mapping:
                    continue

                # 保存RGB图像
                rgb_attr = camera_mapping[camera_name]['rgb']
                rgb_img = getattr(obs, rgb_attr, None)
                if rgb_img is not None:
                    rgb_path = episode_folder.joinpath(camera_name, f"{i}.png")
                    Image.fromarray(rgb_img).save(str(rgb_path))

                # 保存深度图像
                depth_attr = camera_mapping[camera_name]['depth']
                depth_img = getattr(obs, depth_attr, None)
                if depth_img is not None:
                    depth_path = episode_folder.joinpath(f"{camera_name}_depth", f"{i}.png")
                    depth_image = np.clip(depth_img * 100, 0, 255).astype(np.uint8)
                    Image.fromarray(depth_image).save(str(depth_path))

                # 保存掩码图像
                mask_attr = camera_mapping[camera_name]['mask']
                mask_img = getattr(obs, mask_attr, None)

                if mask_img is not None:
                    mask_path = episode_folder.joinpath(f"{camera_name}_mask", f"{i}.png")
                    mask_image = np.clip(mask_img, 0, 255).astype(np.uint8)
                    Image.fromarray(mask_image).save(str(mask_path))
        
        # 保存状态数据到JSON文件
        state_json_path = episode_folder.joinpath("state.json")
        with open(str(state_json_path), 'w') as json_file:
            json.dump(state_data, json_file, indent=4)
        
        print(f"演示 {ex_idx} 原始数据保存成功，路径: {episode_folder}")

    def collect_and_save_demos(self):
        """逐个收集、保存demo，并在每个demo处理完后释放内存"""
        import gc  # 导入垃圾回收模块

        # 创建保存路径
        task_path = os.path.join(self.save_path_head, self.task.get_name())

        if self.save_path_end == "":
            now_time = datetime.datetime.now()
            str_time = now_time.strftime("%Y-%m-%d-%H-%M-%S")
            variation_path = os.path.join(task_path, str_time) 
            self.save_path_end = str_time
        else:
            variation_path = os.path.join(task_path, self.save_path_end)

        self._check_and_make(variation_path)

        # 逐个收集和保存demo
        for i in range(self.num_demos):
            print(f'收集和保存演示 {i}/{self.num_demos}')
            
            # 只获取一个demo
            demo = self.task.get_demos(1, live_demos=True)[0]
            
            # 保存这个demo
            self.save_demo_raw(demo, variation_path, i)
            
            # 显式释放内存
            del demo
            gc.collect()  

        print(f'数据集已保存到 {variation_path}')

    def load_trajectory(self, trajectory_path):
        """
        从指定路径加载轨迹数据
        
        Args:
            trajectory_path: 轨迹数据路径
            
        Returns:
            dict: 加载的轨迹数据
        """
        state_file_path = os.path.join(trajectory_path, 'state.json')
        print(f"加载轨迹数据: {state_file_path}")
        
        with open(state_file_path, 'r') as f:
            trajectory_data = json.load(f)
            
        print(f"成功加载轨迹，共有 {len(trajectory_data)} 个数据点")
        return trajectory_data

    def execute_trajectory(self, trajectory_data, epoch_idx):
        """
        执行轨迹控制
        
        Args:
            trajectory_data: 包含机械臂轨迹的字典
            epoch_idx: 当前epoch的索引
        """
        # 重置任务
        print("重置任务...")
        descriptions, obs = self.task.reset()
        print("任务描述: ", descriptions)
        
        # 按顺序执行轨迹
        print(f"开始执行轨迹 (Epoch {epoch_idx})...")
        trajectory_keys = sorted([int(k) for k in trajectory_data.keys()])
        
        for t in tqdm(trajectory_keys, desc=f"Epoch {epoch_idx} 执行进度"):
            frame_data = trajectory_data[str(t)]
            
            # 提取末端位姿和夹爪动作
            pose = frame_data['robot_state']  # [x, y, z, qx, qy, qz, qw]
            
            # 获取夹爪开合状态 (0-1范围)
            gripper_open = frame_data.get('grasp_state', [0])[0]
            
            # 如果grasp_action不是0-1范围(是0-0.04)，需要归一化
            if gripper_open > 0 and gripper_open <= 0.04:
                gripper_joint_position = gripper_open  # 已经是关节位置
            else:
                # GripperJointPosition需要将0-1范围映射到0-0.04范围
                gripper_joint_position = gripper_open * 0.04
            
            # 执行动作
            action = np.array(pose + [gripper_joint_position])
            obs, reward, terminate = self.task.step(action)
            
            # 简短暂停以便观察
            # time.sleep(0.01)
            
            if terminate:
                print(f"Epoch {epoch_idx} 任务结束")
                break
                
        print(f"轨迹 {epoch_idx} 执行完成")
        
        # 等待一段时间，观察最终状态
        time.sleep(1)

    def process_all_epochs(self):
        """处理所有的epoch（遍历数据文件夹中的所有子文件夹）"""
        print(f"处理来自路径的所有轨迹: {self.data_path}")
        
        # 检查基础路径是否存在
        self._check_path_exists(self.data_path)
        
        # 获取所有子文件夹（epoch）
        epoch_folders = []
        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            # 只处理文件夹且确保有state.json文件
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'state.json')):
                epoch_folders.append(item)
        
        # 确保按照编号排序
        epoch_folders = sorted(epoch_folders, key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        if not epoch_folders:
            print(f"警告: 在 {self.data_path} 中未找到有效的轨迹数据")
            return
            
        print(f"找到 {len(epoch_folders)} 个轨迹")
        
        # 遍历执行每个epoch
        for epoch in epoch_folders:
            epoch_path = os.path.join(self.data_path, epoch)
            print(f"\n处理轨迹: {epoch_path}")
            
            # 加载轨迹
            trajectory_data = self.load_trajectory(epoch_path)
            
            # 执行轨迹
            self.execute_trajectory(trajectory_data, epoch)
            
            # 等待短暂时间，准备下一个轨迹
            time.sleep(0.5)

    def run(self):
        """运行处理器，根据配置决定是数据采样还是轨迹执行"""
        try:
            self.setup_environment()
            self.load_task()
            
            if self.reaction_mode:
                print("以轨迹执行模式运行...")
                self.process_all_epochs()
            else:
                print("以数据采样模式运行...")
                self.collect_and_save_demos()
                
        finally:
            if self.env is not None:
                self.env.shutdown()
                print("环境已关闭")


if __name__ == "__main__":
    # 创建并运行处理器
    processor = RLBenchProcessor()
    processor.run()


