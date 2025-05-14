import numpy as np
import os
from pprint import pprint
import yaml
import importlib
import json
import pathlib
from PIL import Image
import time, datetime
from tqdm import tqdm 
import random
import torch

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import GripperJointPosition
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from act_policy_wrapper import ACTPolicyWrapper


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
        self.save_path_head = self.config.get('save_path_head')
        self.save_path_end = self.config.get('save_path_end')
        self.taskname = self.config.get('taskname')
        self.num_demos = self.config.get('num_demos')
        self.image_width = self.config['image']['width']
        self.image_height = self.config['image']['height']
        self.camera_names = self.config['cameras']
        self.camera_names_forward = self.config['act_policy']['task_config']['camera_names']
        
        # 确定运行模式 (采样 or 反应)
        self.mode = self.config.get('mode', 0)
        print(f"当前模式: {self.mode} (0=采样, 1=轨迹复现, 2=评估)")
        if self.mode == 1:
            self.data_path = self.config['data_path']
        elif self.mode == 2:
            self.act_policy = ACTPolicyWrapper(self.config.get('act_policy'))


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

    def task_file_to_task_class(self, task_file):
        class_name = ''.join([w[0].upper() + w[1:] for w in task_file.split('_')])
        mod = importlib.import_module("rlbench.tasks.%s" % task_file)
        mod = importlib.reload(mod)
        task_class = getattr(mod, class_name)
        return task_class,class_name

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
        if self.mode == 1 or self.mode == 2:
            # 反应模式：使用末端位姿控制
            action_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),
                gripper_action_mode=GripperJointPosition(absolute_mode=True))
            print("使用末端位姿控制模式进行轨迹执行")
        elif self.mode == 0:
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
            # static_positions=True,
        )
            
        self.env.launch()
        print("环境启动成功")

    def load_task(self):
        """加载指定的任务"""
        print(f"加载任务: {self.taskname}")
        task_class,classname = self.task_file_to_task_class(self.taskname)
        print(f"加载任务类: {classname}")
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
                # print("rgb_img.shape",rgb_img.shape)
                if rgb_img is not None:
                    rgb_path = episode_folder.joinpath(camera_name, f"{i}.png")
                    Image.fromarray(rgb_img).save(str(rgb_path))

                # 保存深度图像
                depth_attr = camera_mapping[camera_name]['depth']
                depth_img = getattr(obs, depth_attr, None)
                # print("depth_img.shape",depth_img.shape)
                if depth_img is not None:
                    depth_path = episode_folder.joinpath(f"{camera_name}_depth", f"{i}.png")
                    depth_image = np.clip(depth_img * 100, 0, 255).astype(np.uint8)
                    Image.fromarray(depth_image).save(str(depth_path))

                # 保存掩码图像
                mask_attr = camera_mapping[camera_name]['mask']
                mask_img = getattr(obs, mask_attr, None)
                # print("mask_img.shape",mask_img.shape)

                if mask_img is not None:


                    # 处理掩码图像
                    # 创建空白RGB图像
                    mask_array = mask_img
                    rgb_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
                    
                    # 根据灰度值设置不同的RGB值
                    rgb_array[(mask_array == 35) | (mask_array == 31) | (mask_array == 34) , 0] = 255
                    rgb_array[mask_array == 84, 1] = 255
                    rgb_array[mask_array == 83, 2] = 255

                    mask_path = episode_folder.joinpath(f"{camera_name}_mask", f"{i}.png")
                    mask_image = np.clip(rgb_array, 0, 255).astype(np.uint8)
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
        task_path = os.path.join(self.save_path_head, self.taskname)

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

    def eval_process_observation(self, obs):

        # 提取图像数据
        imgdata = {}


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
            imgdata[camera_name] = rgb_img

            # 保存掩码图像
            mask_attr = camera_mapping[camera_name]['mask']
            mask_img = getattr(obs, mask_attr, None)

            # 处理掩码图像
            # 创建空白RGB图像
            mask_array = mask_img
            rgb_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
            
            # 根据灰度值设置不同的RGB值
            rgb_array[(mask_array == 35) | (mask_array == 31) | (mask_array == 34) , 0] = 255
            rgb_array[mask_array == 84, 1] = 255
            rgb_array[mask_array == 83, 2] = 255

            imgdata[f"{camera_name}_mask"] = rgb_array


        import copy
        # 提取机器人状态
        robot_state = list(obs.joint_positions)  # 关节位置
        robot_state.append(float(1 - obs.gripper_open))  # 夹爪状态（1=关闭，0=打开）
        robot_state = list(copy.deepcopy(obs.gripper_pose))  # 关节位置
        robot_state.append(copy.deepcopy(obs.gripper_open))  # 夹爪状态（1=关闭，0=打开）

        return imgdata, robot_state

    def act_eval(self, max_steps=250, max_attempts=100):
        """
        执行指定任务，失败时自动重试，并统计成功率和平均步骤数
        
        Args:
            max_steps: 每次尝试的最大执行步数
            max_attempts: 最大尝试次数

        Returns:
            tuple: (成功率, 平均步骤数)
        """
        from weight import calculate_change_weight  # 导入计算权重的函数
        
        success_counts = 0  # 成功次数统计
        successful_steps = []  # 记录每次成功尝试的步骤数
        attempt = 0
        
        try:
            while attempt < max_attempts:
                attempt += 1
                print(f"\n开始第 {attempt}/{max_attempts} 次尝试执行任务")
                
                # 重置任务获取初始观察
                descriptions, obs = self.task.reset()
                self.act_policy.reset()
                print(f"任务描述: {descriptions}")
                
                # 用于存储上一帧的图像
                prev_images = None
                
                # 执行控制循环
                success_in_this_attempt = False
                for step in tqdm(range(max_steps), desc=f"第 {attempt} 次尝试"):
                    # 处理观察获取图像和状态
                    imgdata, robot_state = self.eval_process_observation(obs)
                    print(f"机器人状态: {robot_state}")

                    # 计算权重view_weights
                    view_weights = []
                    if prev_images is not None:
                        print("计算视角变化权重：")
                        for cam_name in self.camera_names_forward:
                            if cam_name in imgdata:
                                curr_img = imgdata[f"{cam_name}_mask"]
                                prev_img = prev_images[f"{cam_name}_mask"]
                                # 计算变化权重
                                weight = calculate_change_weight(prev_img, curr_img)
                                view_weights.append(weight)
                                print(f"  - {cam_name}: {weight:.4f}")
                    
                    # 如果有计算出权重，则使用它们；否则使用默认权重
                    if view_weights and len(view_weights) == len(self.camera_names_forward):
                        # 归一化权重，确保总和为相机数量（平均权重为1）
                        total_weight = sum(view_weights)
                        norm_view_weights = [w * len(view_weights) / total_weight for w in view_weights]
                        print(f"归一化视角权重: {[f'{w:.4f}' for w in norm_view_weights]}")
                        actaction = self.act_policy.get_actions(imgdata, robot_state, view_weights=norm_view_weights)
                    else:
                        print("使用默认视角权重")
                        actaction = self.act_policy.get_actions(imgdata, robot_state)

                    # 保存当前帧作为下一次迭代的上一帧
                    # prev_images = {}
                    # for cam_name in self.camera_names_forward:
                    #     if cam_name in imgdata:
                    #         prev_images[f"{cam_name}_mask"] = imgdata[f"{cam_name}_mask"].copy()

                    # 模型输出转换为末端位姿控制
                    end_effector_pose = actaction[0:7].copy()
                    
                    # 单位化四元数
                    quat = end_effector_pose[3:7]
                    norm = np.linalg.norm(quat)
                    if norm < 1e-6:
                        quat = np.array([0.0, 0.0, 0.0, 1.0])
                    else:
                        quat = quat / norm
                    end_effector_pose[3:7] = quat

                    # 夹爪控制
                    gripper_value = actaction[7]
                    gripper_joint_position = 0.04 * float(gripper_value > 0.5)

                    # 执行动作
                    try:
                        action = np.concatenate([end_effector_pose, [gripper_joint_position]]).astype(np.float32)
                        print(f"执行动作: {action}")
                        obs, reward, terminate = self.task.step(action)

                        # 检查任务状态
                        if reward == 1.0:
                            print(f"\n🎉 第 {attempt} 次尝试任务执行成功! 使用步骤: {step+1}")
                            success_counts += 1
                            successful_steps.append(step+1)  # 记录成功所需的步骤数
                            success_in_this_attempt = True
                            break

                        if terminate:
                            print(f"\n❌ 第 {attempt} 次尝试被终止")
                            break

                    except Exception as e:
                        print(f"\n第 {attempt} 次尝试执行动作时发生错误: {e}")
                        break
                
                if not success_in_this_attempt and attempt < max_attempts:
                    print(f"\n第 {attempt} 次尝试未成功，将重新尝试")
                    time.sleep(1)
            
            # 计算统计结果
            success_rate = success_counts / attempt if attempt > 0 else 0
            avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0
            
            print(f"\n===== 评估结果统计 =====")
            print(f"- 总尝试次数: {attempt}")
            print(f"- 成功次数: {success_counts}")
            print(f"- 成功率: {success_rate:.2%}")
            print(f"- 平均成功步骤数: {avg_steps:.2f}")
            if successful_steps:
                print(f"- 最少步骤数: {min(successful_steps)}")
                print(f"- 最多步骤数: {max(successful_steps)}")
            
            return success_rate, avg_steps

        except Exception as e:
            print(f"任务执行过程中发生错误: {e}")
            return 0, 0

        finally:
            # 只在所有尝试结束后关闭环境
            if self.env is not None and attempt >= max_attempts:
                self.env.shutdown()
                print("RLBench环境已关闭")

    def process_all_epochs(self):
        """处理所有的epoch（遍历数据文件夹中的所有子文件夹）"""
        print(f"处理来自路径的所有轨迹: {self.data_path}")
        
        # 检查基础路径是否存在
        self._check_path_exists(self.data_path)
        
        # 获取所有子文件夹（epoch）
        epoch_folders = []
        for item in os.listdir(self.data_path):
            item_path = os.path路径.join(self.data_path, item)
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
            
            if self.mode == 1:
                print("以轨迹复现模式运行...")
                self.process_all_epochs()
            elif self.mode == 0:
                print("以数据采样模式运行...")
                self.collect_and_save_demos()
            elif self.mode == 2:
                print("以评估模式运行...")
                self.act_eval()
                
        finally:
            if self.env is not None:
                self.env.shutdown()
                print("环境已关闭")


if __name__ == "__main__":
    # 创建并运行处理器

    # 在程序开始处添加
    seed = int(time.time()) # 使用当前时间作为种子
    np.random.seed(seed)
    random.seed(seed)
    print(f"使用随机种子: {seed}")

    processor = RLBenchProcessor()
    processor.run()


