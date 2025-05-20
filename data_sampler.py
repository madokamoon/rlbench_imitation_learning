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
import pickle
import gc  
import matplotlib.pyplot as plt
import sys 

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import GripperJointPosition
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from act_policy_wrapper import ACTPolicyWrapper
from pyrep.backend import sim

class RLBenchProcessor:
    """用于RLBench环境的数据采样与轨迹执行的综合处理器"""
    
    def __init__(self, config):

        self.mode = config.get('mode', 0)
        print(f"当前模式: {self.mode} (0=采样, 1=轨迹复现, 2=评估)")

        # 从配置中获取参数
        data_sampler_config = config.get('data_sampler_config', {})
        self.data_sampler_config = data_sampler_config
        self.save_path_head = data_sampler_config['save_path_head']
        self.save_path_end = data_sampler_config['save_path_end']
        self.taskname = data_sampler_config['taskname']
        self.num_demos = data_sampler_config['num_demos']
        self.image_width = data_sampler_config['image']['width']
        self.image_height = data_sampler_config['image']['height']
        self.camera_names = data_sampler_config['cameras']
        self.static_positions = data_sampler_config['static_positions']
        self.headless = data_sampler_config['headless']
        self.robot_init_state = data_sampler_config['robot_init_state']

        # 保存路径
        task_path = os.path.join(self.save_path_head, self.taskname)
        if self.save_path_end == "":
            now_time = datetime.datetime.now()
            str_time = now_time.strftime("%Y-%m-%d-%H-%M-%S")
            self.variation_path = os.path.join(task_path, str_time) 
            self.save_path_end = str_time
        else:
            self.variation_path = os.path.join(task_path, self.save_path_end)
   
        if self.mode == 2:
            act_policy_config = config.get('act_policy')
            self.camera_names_forward = act_policy_config['task_config']['camera_names']
            self.use_weight = act_policy_config['use_weight']
            self.max_steps =  act_policy_config['task_config']['episode_len']
            self.act_policy = ACTPolicyWrapper(act_policy_config)


        self.env = None
        self.task = None
        self.obs_config = None

    def _check_and_make(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
        else :
            user_input = input("输出路径已经存在，是否覆盖？(y/n): ")
            if user_input.lower() == 'y':
                import shutil
                shutil.rmtree(directory)
                print(f"已删除现有目录: {directory}")
                os.makedirs(directory)
                print(f"创建目录: {directory}")
            else:
                print("操作已取消")
                sys.exit(0) 
                return

            
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
        obs_config.gripper_touch_forces = True 
        # obs_config.record_gripper_closing = True  # 会卡很久
          
        return obs_config
    
        # 更多ObservationConfig 可配置参数：

        # | 属性名                       | 类型             | 说明                  |
        # | ------------------------- | -------------- | ------------------- |
        # | `front_camera`            | `CameraConfig` | 前置摄像头配置（RGB/深度/分割等） |
        # | `left_shoulder_camera`    | `CameraConfig` | 左肩摄像头配置             |
        # | `right_shoulder_camera`   | `CameraConfig` | 右肩摄像头配置             |
        # | `overhead_camera`         | `CameraConfig` | 俯视摄像头配置             |
        # | `wrist_camera`            | `CameraConfig` | 手腕摄像头配置             |
        # | `wrist_camera_matrix`     | `bool`         | 是否返回手腕摄像头的 4x4 变换矩阵 |
        # | `gripper_open`            | `bool`         | 返回夹爪是否张开（开/合）       |
        # | `gripper_pose`            | `bool`         | 返回夹爪的世界位姿（位置 + 方向）  |
        # | `gripper_joint_positions` | `bool`         | 返回夹爪的各个关节角度         |
        # | `gripper_matrix`          | `bool`         | 返回夹爪的 4x4 变换矩阵      |
        # | `gripper_touch_forces`    | `bool`         | 返回夹爪的触觉接触力量         |
        # | `joint_positions`         | `bool`         | 返回机械臂的各个关节位置        |
        # | `joint_velocities`        | `bool`         | 返回机械臂的各个关节速度        |
        # | `joint_forces`            | `bool`         | 返回机械臂的各个关节受力（力矩）    |
        # | `joint_positions_noise`   | `NoiseModel`   | 对关节位置加入噪声的模型        |
        # | `joint_velocities_noise`  | `NoiseModel`   | 对关节速度加入噪声的模型        |
        # | `joint_forces_noise`      | `NoiseModel`   | 对关节力加入噪声的模型         |
        # | `task_low_dim_state`      | `bool`         | 是否返回任务的低维状态（手工特征）   |
        # | `record_gripper_closing`  | `bool`         | 是否记录夹爪闭合过程（用于数据集生成） |




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
        
        if self.mode == 1 or self.mode == 2:
            # 反应模式：使用末端位姿控制
            action_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),
                gripper_action_mode=GripperJointPosition(absolute_mode=True))
            print("末端位姿控制模式")
        elif self.mode == 0 :
            # 采样模式：使用关节速度控制
            action_mode = MoveArmThenGripper(
                arm_action_mode=JointVelocity(),
                gripper_action_mode=GripperJointPosition(absolute_mode=True))
            print("关节速度控制模式")
            
        # 创建并配置RLBench环境
        self.env = Environment(
            action_mode=action_mode,
            obs_config=self.obs_config, 
            headless=self.headless,
        )
            
        # 使用sim模块直接设置物理引擎参数 
        # sim.simSetEngineFloatParameter()
        # sim.simSetEngineFloatParameter()


        self.env.launch()
        print("rlbench环境启动成功")

    def load_task(self):
        """加载指定的任务"""
        print(f"加载任务: {self.taskname}")
        task_class,classname = self.task_file_to_task_class(self.taskname)
        print(f"加载任务类: {classname}")
        self.task = self.env.get_task(task_class)

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
    
        self._check_and_make(self.variation_path)

        config_save_path = os.path.join(self.variation_path, "data_sampler_config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(self.data_sampler_config, f, default_flow_style=False)
    
        print(f"本次运行配置已保存到: {config_save_path}")

        if self.static_positions == True:

            print("静态位置模式")
            # 环境状态保存路径
            env_state_path = os.path.join(self.variation_path, "initial_state.pickle")
            descriptions, first_obs = self.task.reset()
            random_state = random.getstate()
            numpy_state = np.random.get_state()
            reference_demo = self.task.get_demos(1, live_demos=True)[0]
            # 只保留第一个观察
            if len(reference_demo._observations) > 1:
                first_obs = reference_demo._observations[0]
                reference_demo._observations = [first_obs]
            # 保存环境信息
            env_info = {
                'descriptions': descriptions,
                'random_state': random_state,
                'numpy_state': numpy_state,
                'reference_demo': reference_demo,  # 保存参考演示对象
                'timestamp': datetime.datetime.now().isoformat()
            }
            with open(env_state_path, 'wb') as f:
                pickle.dump(env_info, f)
            
            print(f"初始环境配置已保存到: {env_state_path}")

            del reference_demo
            gc.collect() 

            reference_demo, random_state, numpy_state = self.load_reference_demo()

        # 逐个收集和保存demo
        for i in range(self.num_demos):
            print(f'收集和保存演示 {i}/{self.num_demos}')
            
            if self.static_positions == True:
                random.setstate(random_state)
                np.random.set_state(numpy_state)
                self.task.reset_to_demo(reference_demo)
                # 收集的时候不能加上这句，其他情况需要加上，原因为 task.get_demos 中会调用 reset()
                # descriptions, obs = self.task.reset()  
            else:
                pass
                # descriptions, obs = self.task.reset()

            if self.robot_init_state is not None:
                    self.task.step(self.robot_init_state)

            # 只获取一个demo
            demo = self.task.get_demos(1, live_demos=True)[0]
            
            # 保存这个demo
            self.save_demo_raw(demo, self.variation_path, i)
            
            # 显式释放内存
            del demo
            gc.collect()  

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

        # 按顺序执行轨迹
        print(f"开始执行轨迹 (Epoch {epoch_idx})...")
        trajectory_keys = sorted([int(k) for k in trajectory_data.keys()])
        
        for t in tqdm(trajectory_keys, desc=f"Epoch {epoch_idx} 执行进度"):

            frame_data = trajectory_data[str(t)]
            
            # 提取末端位姿和夹爪动作
            pose = frame_data['robot_state']  # [x, y, z, qx, qy, qz, qw]
            
            # 获取夹爪开合状态 
            gripper_open = frame_data.get('grasp_state', [0])[0]
            gripper_joint_position = 0.04 * float(gripper_open > 0.5) 


            # 执行动作.
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

        # 关节控制
        # robot_state = list(copy.deepcopy(obs.joint_positions)) 
        # robot_state.append(copy.deepcopy(obs.gripper_open))  

        # 末端位姿控制
        robot_state = list(copy.deepcopy(obs.gripper_pose)) 
        robot_state.append(copy.deepcopy(obs.gripper_open)) 

        return imgdata, robot_state

    def load_reference_demo(self):   

        reference_demo = None
        random_state = None
        numpy_state = None

        env_state_path = os.path.join(self.variation_path, "initial_state.pickle")
        if env_state_path and os.path.exists(env_state_path):
            print(f"加载环境状态数据: {env_state_path}")
            try:
                with open(env_state_path, 'rb') as f:
                    env_info = pickle.load(f)
            
                # 获取存储的状态
                random_state = env_info.get('random_state')
                numpy_state = env_info.get('numpy_state')
                reference_demo = env_info.get('reference_demo')  # 直接获取保存的参考演示
                descriptions = env_info.get('descriptions')
                
                print(f"成功加载环境状态，描述: {descriptions}")
                return reference_demo,random_state,numpy_state
        
            except Exception as e:
                print(f"加载环境状态失败: {e}")
                env_state_path = None
                return None


    def act_eval(self, max_attempts=20):
        """
        执行指定任务，失败时自动重试，并统计成功率和平均步骤数
        
        Args:
            max_attempts: 最大尝试次数

        Returns:
            tuple: (成功率, 平均步骤数)
        """
        import traceback  # 导入traceback模块
        from weight.weight import calculate_change_weight

        success_counts = 0  # 成功次数统计
        successful_steps = []  # 记录每次成功尝试的步骤数
        attempt = 0
        
        if self.static_positions == True:
            reference_demo,random_state,numpy_state = self.load_reference_demo()

        try:
            while attempt < max_attempts:
                attempt += 1
                print(f"\n开始第 {attempt}/{max_attempts} 次尝试执行任务")
                
                if self.static_positions == True:
                    random.setstate(random_state)
                    np.random.set_state(numpy_state)
                    self.task.reset_to_demo(reference_demo)
                    descriptions, obs = self.task.reset()


                else:
                    descriptions, obs = self.task.reset()


                self.act_policy.reset()
                # 用于存储上一帧的图像
                prev_images = None
                
                # 执行控制循环
                success_in_this_attempt = False

                # forceslist = []

                for step in tqdm(range(self.max_steps), desc=f"第 {attempt} 次尝试"):
                    # 处理观察获取图像和状态
                    imgdata, robot_state = self.eval_process_observation(obs)
                    formatted_state = [f"{val:8.5f}" for val in robot_state]
                    print(f"robot_state_:{formatted_state}")

                    if self.use_weight :
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
                            
                            # 归一化权重，确保总和为相机数量（平均权重为1）
                            total_weight = sum(view_weights)
                            norm_view_weights = [w * len(view_weights) / total_weight for w in view_weights]
                            print(f"归一化视角权重: {[f'{w:.4f}' for w in norm_view_weights]}")
                            actaction = self.act_policy.get_actions(imgdata, robot_state, view_weights=norm_view_weights)
                        else:
                            print("没有上一帧图像，使用默认权重")
                            actaction = self.act_policy.get_actions(imgdata, robot_state)
                            
                        # 保存当前帧作为下一次迭代的上一帧
                        prev_images = {}
                        for cam_name in self.camera_names_forward:
                            if cam_name in imgdata:
                                prev_images[f"{cam_name}_mask"] = imgdata[f"{cam_name}_mask"].copy()

                    else:
                        actaction = self.act_policy.get_actions(imgdata, robot_state)
                        

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
                        formatted_action = [f"{val:8.5f}" for val in action]
                        print(f"robot_action:{formatted_action}")
                        obs, reward, terminate = self.task.step(action)

                        # forceslist.append(obs.gripper_touch_forces)
                        # formatted_forces = [f"{val:8.5f}" for val in obs.gripper_touch_forces]
                        # print(f"touch_forces:{formatted_forces}")



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
                        print(f"\n第 {attempt} 次尝试执行动作时发生错误:")
                        traceback.print_exc()  # 打印完整的堆栈跟踪
                        break
                
                # forces_array = np.array(forceslist)  
                # 获取力传感器数量
                # num_sensors = forces_array.shape[1]
                # time_steps = forces_array.shape[0]
                # plt.figure(figsize=(12, 6))
                # for i in range(num_sensors):
                #     plt.plot(range(time_steps), forces_array[:, i], label=f'force {i+1}')
                # plt.grid(True)
                # plt.legend()
                # plt.show()

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
            print(f"任务执行过程中发生错误:")
            traceback.print_exc()  # 打印完整的堆栈跟踪
            return 0, 0

        finally:
            # 只在所有尝试结束后关闭环境
            if self.env is not None and attempt >= max_attempts:
                self.env.shutdown()
                print("RLBench环境已关闭")

    def process_all_epochs(self):
        """处理所有的epoch（遍历数据文件夹中的所有子文件夹）"""
        print(f"处理来自路径的所有轨迹: {self.variation_path}")
        
        # 检查基础路径是否存在
        self._check_path_exists(self.variation_path)
        
        # 获取所有子文件夹（epoch）
        epoch_folders = []
        for item in os.listdir(self.variation_path):
            item_path = os.path.join(self.variation_path, item)
            # 只处理文件夹且确保有state.json文件
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'state.json')):
                epoch_folders.append(item)
        
        # 确保按照编号排序
        epoch_folders = sorted(epoch_folders, key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        if not epoch_folders:
            print(f"警告: 在 {self.variation_path} 中未找到有效的轨迹数据")
            return
            
        print(f"找到 {len(epoch_folders)} 个轨迹")
        

        if self.static_positions == True:
            reference_demo, random_state, numpy_state = self.load_reference_demo()

        # 遍历执行每个epoch
        for epoch in epoch_folders:
            epoch_path = os.path.join(self.variation_path, epoch)
            print(f"\n处理轨迹: {epoch_path}")
            
            # 加载轨迹
            trajectory_data = self.load_trajectory(epoch_path)
            
            if self.static_positions == True:
                random.setstate(random_state)
                np.random.set_state(numpy_state)
                self.task.reset_to_demo(reference_demo)
                descriptions, obs = self.task.reset()
            else:
                descriptions, obs = self.task.reset()

            # 执行轨迹
            self.execute_trajectory(trajectory_data, epoch)
            
            # 等待短暂时间，准备下一个轨迹
            time.sleep(0.5)

    def run(self):
        """运行处理器，根据配置决定是数据采样还是轨迹执行"""
        try:
            self.setup_environment()
            self.load_task()
            
            if self.mode == 0:
                print("以数据采样模式运行...")
                self.collect_and_save_demos()
            elif self.mode == 1:
                print("以轨迹复现模式运行...")
                self.process_all_epochs()
            elif self.mode == 2:
                print("以评估模式运行...")
                self.act_eval()
                
        finally:
            if self.env is not None:
                self.env.shutdown()
                print("环境已关闭")




if __name__ == "__main__":

    # 使用当前时间作为种子
    # seed = int(time.time()) 
    # np.random.seed(seed)
    # random.seed(seed)
    # print(f"使用随机种子: {seed}")'


    import argparse
    parser = argparse.ArgumentParser(description='RLBench数据采样与轨迹执行工具')
    parser.add_argument('--config', type=str, default=None, help='指定配置文件路径')
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            print(f"使用命令行配置文件: {args.config}")
            config = yaml.safe_load(f)
    elif os.path.exists('data_sampler_local.yaml'):
        with open('data_sampler_local.yaml', 'r') as f:
            print("使用本地配置文件 data_sampler_local.yaml")
            config = yaml.safe_load(f)
    else:
        with open('data_sampler.yaml', 'r') as f:
            print("使用默认配置文件 data_sampler.yaml")
            config = yaml.safe_load(f)

    processor = RLBenchProcessor(config)
    processor.run()


