import importlib
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from pprint import pprint
import argparse
import pathlib
import matplotlib.pyplot as plt
import os
import gc

# RLBench相关模块
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import GripperJointPosition
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

# 导入ACT策略封装类
from act_policy_wrapper import ACTPolicyWrapper


class RLBenchACTController:
    """
    使用ACT策略控制RLBench环境中机器人的控制器类
    """
    
    def __init__(self, config_path='data_sampler.yaml'):
        """
        初始化控制器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 从配置中获取参数
        self.taskname = self.config.get('taskname')
        self.max_steps = self.config.get('max_steps', 1000)
        self.image_width = self.config['image']['width']
        self.image_height = self.config['image']['height']
        self.camera_names = self.config['cameras']
        
        # 初始化ACT策略封装
        self.act_policy = ACTPolicyWrapper(self.config.get('act_policy'))
        
        # 存储环境和任务相关变量
        self.env = None
        self.task = None
        # 移除特定机器人设置，使用默认设置
        self.robot_setup = None
        
        # 用于保存关节位置数据的字典
        self.joint_data = {
            'current': [],  # 当前关节位置
            'target': []    # 目标关节位置
        }
    
    def create_observation_config(self):
        """创建RLBench观察配置"""
        obs_config = ObservationConfig()
        
        # 相机配置映射
        cameras_dict = {
            'front_camera': obs_config.front_camera,
            'wrist_camera': obs_config.wrist_camera,
            'left_shoulder_camera': obs_config.left_shoulder_camera,
            'right_shoulder_camera': obs_config.right_shoulder_camera,
            'overhead_camera': obs_config.overhead_camera
        }
        
        # 先禁用所有相机
        for camera in cameras_dict.values():
            camera.rgb = False
            
        # 启用指定的相机
        for camera_name in self.camera_names:
            if camera_name in cameras_dict:
                cameras_dict[camera_name].rgb = True
                cameras_dict[camera_name].image_size = (self.image_width, self.image_height)
                print(f"启用相机: {camera_name}")
        
        # 启用必要的传感器
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.gripper_open = True
        obs_config.gripper_pose = True  # 启用末端位姿
        obs_config.gripper_joint_positions = True
        obs_config.gripper_matrix = True
        
        return obs_config
    
    def setup_environment(self):
        """设置并启动RLBench环境"""
        # 创建观察配置
        obs_config = self.create_observation_config()
        
        # 打印配置信息
        print("\n观察配置属性:")
        pprint(obs_config.__dict__)
        print("\n")
        
        # 创建环境 - 修改为末端位姿控制和夹爪位置控制
        self.env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),  # 改为末端位姿控制
                gripper_action_mode=GripperJointPosition(absolute_mode=True)), 
            obs_config=obs_config, 
            headless=False)  # 使用默认机器人设置
        
        # 启动环境
        self.env.launch()
        print("RLBench环境已启动")
    
    def task_file_to_task_class(self, task_file):
        """将任务文件名转换为任务类"""
        class_name = ''.join([w[0].upper() + w[1:] for w in task_file.split('_')])
        mod = importlib.import_module("rlbench.tasks.%s" % task_file)
        mod = importlib.reload(mod)
        task_class = getattr(mod, class_name)
        return task_class, class_name
    
    def load_task(self, task_name):
        """
        加载指定的任务
        
        Args:
            task_name: 任务类名称
        """
        try:
            task_class, class_name = self.task_file_to_task_class(task_name)
            self.task = self.env.get_task(task_class)
            print(f"已加载任务: {task_name} (类名: {class_name})")
            return True
        except (ImportError, AttributeError) as e:
            print(f"加载任务失败: {e}")
            return False
    
    def process_observation(self, obs):
        """
        处理RLBench观察对象为ACT模型可用的格式
        
        Args:
            obs: RLBench观察对象
            
        Returns:
            imgdata: 包含图像的字典
            robot_state: 机器人状态（关节位置和夹爪状态）
        """
        # 提取图像数据
        imgdata = {}
        
        # 使用字典映射相机属性名到观测对象的属性
        camera_mapping = {
            'wrist_camera': 'wrist_rgb',
            'front_camera': 'front_rgb',
            'left_shoulder_camera': 'left_shoulder_rgb',
            'right_shoulder_camera': 'right_shoulder_rgb',
            'overhead_camera': 'overhead_rgb'
        }
        
        # 遍历所有相机
        for camera_name in self.camera_names:
            if camera_name in camera_mapping:
                rgb_attr = camera_mapping[camera_name]
                rgb_img = getattr(obs, rgb_attr, None)
                if rgb_img is not None:
                    imgdata[camera_name] = rgb_img
        
        import copy
        # 提取机器人状态
        robot_state = list(copy.deepcopy(obs.gripper_pose))  # 关节位置
        robot_state.append(copy.deepcopy(obs.gripper_open))  # 夹爪状态（1=关闭，0=打开）
        
        return imgdata, robot_state
    
    def run_task(self, max_steps=None):
        """
        执行指定任务
        
        Args:
            max_steps: 最大执行步数
            
        Returns:
            success: 任务是否成功完成
        """
        if max_steps is None:
            max_steps = self.max_steps
            
        try:
            # 设置环境
            self.setup_environment()
            
            # 加载任务
            if not self.load_task(self.taskname):
                return False
            
            # 重置任务获取初始观察
            descriptions, obs = self.task.reset()
            print(f"任务描述: {descriptions}")
            
            # 执行控制循环
            success = False
            for step in tqdm(range(max_steps), desc="任务执行"):
                # 处理观察获取图像和状态
                imgdata, robot_state = self.process_observation(obs)
                
                # 使用ACT模型获取动作
                actaction = self.act_policy.get_actions(imgdata, robot_state)
                
                # 模型输出转换为末端位姿控制
                # 模型输出的前7个值作为位置和四元数 [x, y, z, qx, qy, qz, qw]
                end_effector_pose = actaction[0:7].copy()  # 避免原地修改 actaction

                # 单位化四元数
                quat = end_effector_pose[3:7]
                norm = np.linalg.norm(quat)
                if norm < 1e-6:
                    # 默认使用单位四元数，避免除以零
                    quat = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    quat = quat / norm
                end_effector_pose[3:7] = quat

                # 夹爪控制 - 从二值转换为连续值
                gripper_value = actaction[7]
                
                # 将夹爪值转换为关节位置 (0-0.04范围)
                # 如果模型输出是二值的，大于0.5表示关闭(接近0.04)，小于0.5表示打开(接近0)
                gripper_joint_position = 0.04 * float(gripper_value > 0.5)
                
                # 执行动作
                try:
                    # 合并末端位姿和夹爪关节位置
                    action = np.concatenate([end_effector_pose, [gripper_joint_position]]).astype(np.float32)
                    obs, reward, terminate = self.task.step(action)

                    # 检查任务是否成功完成
                    if reward == 1.0:
                        print("\n🎉 任务执行成功!")
                        success = True
                        break
                    
                    if terminate:
                        print("\n❌ 任务被终止")
                        break
                        
                except Exception as e:
                    print(f"\n执行动作时发生错误: {e}")
                    break
            
            print(f"\n任务执行结束. {'成功' if success else '未成功'}")
            return success
            
        except Exception as e:
            print(f"任务执行过程中发生错误: {e}")
            return False
        
        finally:
            # 关闭环境
            if self.env is not None:
                self.env.shutdown()
                print("RLBench环境已关闭")
    
    def cleanup_memory(self):
        """
        清理内存，特别是在多次任务执行之间
        """
        if self.env is not None:
            self.env.shutdown()
            self.env = None
            self.task = None
        
        # 强制垃圾回收
        gc.collect()
        print("内存已清理")


def main():
    """主函数，从配置文件加载所有参数"""
    rootpath = pathlib.Path(__file__).parent
    default_config = rootpath.joinpath('data_sampler.yaml')
    
    parser = argparse.ArgumentParser(description='ACT RLBench控制器')
    parser.add_argument('--config', type=str, default=default_config,
                        help='配置文件路径')
    args = parser.parse_args()
    
    # 创建控制器
    controller = RLBenchACTController(args.config)
    
    # 执行任务
    success = controller.run_task()
    
    # 清理内存
    controller.cleanup_memory()
    
    # 返回结果代码
    return 0 if success else 1


if __name__ == "__main__":
    main()