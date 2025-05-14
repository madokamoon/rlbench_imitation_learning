import pathlib
import json
import copy
import numpy as np
import yaml
import imageio.v2 as imageio
import concurrent.futures
import tqdm
import os
import h5py
from PIL import Image
import glob

# 导入权重计算函数
from weight import calculate_change_weight

class RawToHDF5Converter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.camera_names = []
        self.datas = {}
        self.record_path = ""
        self.prev_frames = {}  # 存储前一帧的图像
        
    def convert(self):
        folders = [f for f in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, f))]
        if not folders:
            print("输入路径下没有文件夹")
            return
        
        print(f"找到以下文件夹: {folders}")
        print(f"共计 {len(folders)} 个文件夹需要处理")
        
        # 处理每个文件夹
        for folder_idx, folder_name in enumerate(folders):
            print(f"\n---处理文件夹 {folder_idx+1}/{len(folders)}: {folder_name} ---")
            folder_path = os.path.join(self.input_path, folder_name)
            
            # 重置数据结构
            self.datas = {}
            
            self.camera_names = []

            for f in os.listdir(folder_path):
                if os.path.isdir(os.path.join(folder_path, f)):
                    if f.endswith('depth'):
                        pass
                        # self.camera_names_depth.append(f)
                    elif f.endswith('mask'):
                        self.camera_names.append(f)
                    elif f.endswith('camera'):
                        self.camera_names.append(f)
                        
            
            # 处理深度摄像头图像
            # self.process_depth_images(folder_path)
            # 处理掩码摄像头图像
            # self.process_mask_images(folder_path)

            
            # 初始化数据字典
            self.datas = {
                '/observations/qpos': [],
                '/action': [],
                # 添加三个新的数据项
                '/observations/robot_joint_state': [],
                '/observations/robot_joint_vel': [],
                '/robot_joint_action': [],
            }
            
            # 为每个摄像头创建数据项和权重项
            for cam_name in self.camera_names:
                self.datas[f'/observations/images/{cam_name}'] = []
                self.datas[f'/observations/weight/{cam_name}'] = []  # 初始化权重列表
            
            # 重置前一帧图像
            self.prev_frames = {}
            
            # 读取状态文件
            json_path = os.path.join(folder_path, 'state.json')
            if not os.path.exists(json_path):
                print(f"找不到状态文件: {json_path}")
                continue
            
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # 检查帧数量是否匹配
            frame_count = len(json_data)
            if not self.camera_names:
                print("警告: 没有找到摄像头文件夹，跳过此文件夹")
                continue
                
            camera_frame_count = len(glob.glob(os.path.join(folder_path, self.camera_names[0], '*.png')))
            
            if frame_count != camera_frame_count:
                print(f"警告: JSON文件中的帧数 ({frame_count}) 与摄像头文件夹中的帧数 ({camera_frame_count}) 不匹配")
                continue
            
            print(f"处理 {frame_count} 帧数据...")
            
            # 在处理每一帧数据的循环中，修改权重计算部分
            for frame_idx_str in tqdm.tqdm(sorted(json_data.keys(), key=lambda x: int(x)), 
                                     desc=f"处理文件夹 {folder_name}", 
                                     total=frame_count):
                jsondata = json_data[frame_idx_str]
                frame_idx = int(frame_idx_str)
                
                # 添加机器人状态和抓取状态
                self.datas['/observations/qpos'].append(jsondata["robot_state"] + [jsondata['grasp_state'][0]])
                self.datas['/action'].append(jsondata['robot_action'] + [jsondata['grasp_action'][0]])
                self.datas['/observations/robot_joint_state'].append(jsondata["robot_joint_state"] + [jsondata['grasp_state'][0]])
                self.datas['/observations/robot_joint_vel'].append(jsondata["robot_joint_vel"])
                self.datas['/robot_joint_action'].append(jsondata["robot_joint_action"] + [jsondata['grasp_action'][0]]) 
                
                # 临时存储该帧所有摄像头的原始权重
                frame_weights = {}
                
                # 处理每个摄像头的图像
                for cam_name in self.camera_names:
                    img_path = os.path.join(folder_path, cam_name, f"{frame_idx}.png")
                    if not os.path.exists(img_path):
                        print(f"找不到图像: {img_path}")
                        continue
                    
                    # 读取并处理图像
                    img = Image.open(img_path)
                    
                    # 检查图像尺寸，如果不是480x640就调整
                    if img.size != (640, 480):
                        img = img.resize((640, 480))
                    
                    # 转换为numpy数组
                    img_array = np.array(img)
                    self.datas[f'/observations/images/{cam_name}'].append(img_array)
                    
                    # 计算权重（与前一帧比较）
                    if frame_idx > 0 and cam_name in self.prev_frames:
                        # 计算当前帧与前一帧的变化权重
                        weight = calculate_change_weight(self.prev_frames[cam_name], img_array)
                    else:
                        # 第一帧或无前一帧数据时，设置默认权重1.0
                        weight = 1.0
                    
                    # 暂存权重
                    frame_weights[cam_name] = weight
                    
                    # 更新前一帧图像
                    self.prev_frames[cam_name] = img_array.copy()
                
                # 归一化权重，确保所有摄像头的平均权重为1
                if frame_weights:
                    # 将摄像头分为两组
                    mask_cams = {cam: weight for cam, weight in frame_weights.items() if cam.endswith('mask')}
                    camera_cams = {cam: weight for cam, weight in frame_weights.items() if cam.endswith('camera')}
                    
                    # 处理mask摄像头的权重 - 进行归一化
                    if mask_cams:
                        total_mask_weight = sum(mask_cams.values())
                        mask_count = len(mask_cams)
                        
                        if total_mask_weight > 0:  # 避免除以零
                            # 归一化公式：weight * 摄像头数量 / 总权重
                            for cam_name, weight in mask_cams.items():
                                norm_weight = weight * mask_count / total_mask_weight
                                self.datas[f'/observations/weight/{cam_name}'].append(norm_weight)
                        else:
                            # 如果总权重为零，所有mask摄像头权重设为1
                            for cam_name in mask_cams:
                                self.datas[f'/observations/weight/{cam_name}'].append(1.0)

                    # 处理camera摄像头的权重 - 全部设置为1
                    for cam_name in camera_cams:
                        self.datas[f'/observations/weight/{cam_name}'].append(1.0)

            
            # 设置输出文件路径
            self.record_path = os.path.join(self.output_path, f"episode_{folder_name}.hdf5")
            
            # 保存为HDF5文件
            print(f"正在保存数据到HDF5文件...")
            self.save_to_hdf5()
            
            print(f"已保存到: {self.record_path}")
    
    def save_to_hdf5(self):
        # 获取时间步长
        max_timesteps = len(self.datas['/action'])
        
        with h5py.File(self.record_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            # 设置全局参数
            root.attrs['sim'] = False
            
            # 创建观测组
            obs = root.create_group('observations')
            
            # 创建图像组
            image = obs.create_group('images')
            print("self.camera_names",self.camera_names)
            for cam_name in self.camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), 
                                        dtype='uint8', chunks=(1, 480, 640, 3))
                
            # 创建权重组
            weight = obs.create_group('weight')
            
            # 创建权重数据集
            for cam_name in self.camera_names:
                _ = weight.create_dataset(cam_name, (max_timesteps,), dtype='float32')
        
            # 修改维度
            qpos = obs.create_dataset('qpos', (max_timesteps, 8))
            qvel = obs.create_dataset('qvel', (max_timesteps, 8))
            action = root.create_dataset('action', (max_timesteps, 8))
            
            # 创建三个新的数据集
            if '/observations/robot_joint_state' in self.datas and self.datas['/observations/robot_joint_state']:
                joint_dim = len(self.datas['/observations/robot_joint_state'][0])
                robot_joint_state = obs.create_dataset('robot_joint_state', (max_timesteps, joint_dim))
            
            if '/observations/robot_joint_vel' in self.datas and self.datas['/observations/robot_joint_vel']:
                joint_vel_dim = len(self.datas['/observations/robot_joint_vel'][0])
                robot_joint_vel = obs.create_dataset('robot_joint_vel', (max_timesteps, joint_vel_dim))
            
            if '/robot_joint_action' in self.datas and self.datas['/robot_joint_action']:
                joint_action_dim = len(self.datas['/robot_joint_action'][0])
                robot_joint_action = root.create_dataset('robot_joint_action', (max_timesteps, joint_action_dim))
            
            # 存入数据
            for name, array in self.datas.items():
                if array:  # 只处理非空数据
                    root[name][...] = np.array(array)

    def process_depth_images(self, folder_path):
        """
        处理深度图像，将其转换为RGB格式并保存到新文件夹
        """
        for depth_cam in self.camera_names_depth:
            # 创建新文件夹名
            new_folder_name = depth_cam + "_2"
            new_folder_path = os.path.join(folder_path, new_folder_name)
            
            # 创建新文件夹
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"创建文件夹: {new_folder_path}")
            
            # 处理所有深度图像
            depth_folder_path = os.path.join(folder_path, depth_cam)
            img_files = [f for f in os.listdir(depth_folder_path) if f.endswith('.png')]
            
            print(f"正在处理深度摄像头 {depth_cam} 的 {len(img_files)} 张图像...")
            
            for img_file in tqdm.tqdm(img_files, desc=f"处理深度图像 {depth_cam}"):
                img_path = os.path.join(depth_folder_path, img_file)
                
                # 读取深度图像为灰度图
                depth_img = Image.open(img_path).convert('L')
                
                # 检查图像尺寸，如果不是480x640就调整
                if depth_img.size != (640, 480):
                    depth_img = depth_img.resize((640, 480))
                
                # 将灰度图转换为3通道RGB图像
                rgb_img = Image.merge('RGB', [depth_img, depth_img, depth_img])
                
                # 保存到新文件夹
                rgb_img.save(os.path.join(new_folder_path, img_file))
            
            # 将新文件夹添加到普通摄像头列表，这样会自动将其包含在HDF5输出中
            self.camera_names.append(new_folder_name)

    def process_mask_images(self, folder_path):
        """
        处理掩码图像，将其转换为RGB格式并保存到新文件夹
        """
        for mask_cam in self.camera_names_mask:
            # 创建新文件夹名
            new_folder_name = mask_cam + "_rgb"
            new_folder_path = os.path.join(folder_path, new_folder_name)
            
            # 创建新文件夹
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"创建文件夹: {new_folder_path}")
            
            # 处理所有掩码图像
            mask_folder_path = os.path.join(folder_path, mask_cam)
            img_files = [f for f in os.listdir(mask_folder_path) if f.endswith('.png')]
            
            for img_file in tqdm.tqdm(img_files, desc=f"处理掩码图像 {mask_cam}"):
                img_path = os.path.join(mask_folder_path, img_file)
                
                # 读取掩码图像为灰度图
                mask_img = Image.open(img_path).convert('L')
                
                # 检查图像尺寸，如果不是480x640就调整
                if mask_img.size != (640, 480):
                    mask_img = mask_img.resize((640, 480))
                
                # 将灰度图转换为numpy数组以便处理
                mask_array = np.array(mask_img)
                
                # 创建空白RGB图像
                rgb_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
                
                # 根据灰度值设置不同的RGB值
                rgb_array[(mask_array == 35) | (mask_array == 31) | (mask_array == 34) , 0] = 255
                rgb_array[mask_array == 84, 1] = 255
                rgb_array[mask_array == 83, 2] = 255

                # 转换为PIL图像
                rgb_img = Image.fromarray(rgb_array)
                
                # 保存到新文件夹
                rgb_img.save(os.path.join(new_folder_path, img_file))
            
            # 将新文件夹添加到普通摄像头列表，这样会自动将其包含在HDF5输出中
            self.camera_names.append(new_folder_name)


def main(config_path='data_sampler.yaml'):

        # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 从配置中获取参数
    src_path = config.get('src_path')

    rootpath = pathlib.Path(__file__).parent
    input_path = rootpath.joinpath(src_path) 
    output_path = input_path.parent.joinpath(input_path.name + "_hdf5")

    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入路径不存在: {input_path}")
        return
    # 检查输出路径是否存在
    if os.path.exists(output_path):
        user_input = input("输出路径已经存在，是否覆盖？(y/n): ")
        if user_input.lower() == 'y':
            import shutil
            shutil.rmtree(output_path)
            print(f"已删除现有目录: {output_path}")
        elif user_input.lower() == 'n':
            print("操作已取消")
            return
        else:
            print("无效输入，操作已取消")
            return
    
    output_path.mkdir(parents=True, exist_ok=True)
    print("----------开始转换---------")
    print("根目录：", rootpath)
    print("输入路径：", input_path)
    print("输出路径：", output_path)
    converter = RawToHDF5Converter(input_path, output_path)
    converter.convert()
    print("----------转换完成----------")



if __name__ == '__main__':
    main()
