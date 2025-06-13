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
import time, datetime
import hydra
from omegaconf import OmegaConf
import cv2
import matplotlib.pyplot as plt

# 导入权重计算函数


OmegaConf.register_new_resolver("eval", eval, replace=True)

class RawToHDF5Converter:
    def __init__(self, input_path, output_path, image_width=640, image_height=480,taskname=None,cameraclass=[]):
        self.input_path = input_path
        self.output_path = output_path
        self.camera_names = []
        self.datas = {}
        self.record_path = ""
        self.image_width = image_width
        self.image_height = image_height
        self.taskname = taskname
        self.cameraclass = cameraclass
        
    def convert(self, max_workers=None):
        folders = [f for f in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, f))]
        folders = sorted(folders, key=int)  # 按照数字大小排序
        if not folders:
            print("输入路径下没有文件夹")
            return
        
        print(f"找到以下文件夹: {folders}")
        print(f"共计 {len(folders)} 个文件夹需要处理")
        
        # 提前检测相机文件夹
        self.camera_names = []
        # 使用第一个文件夹作为参考来检测相机文件夹
        if folders:
            folder_path = os.path.join(self.input_path, folders[0])

            for end in self.cameraclass:
                for f in os.listdir(folder_path):
                    if os.path.isdir(os.path.join(folder_path, f)):
                            if f.endswith(end):
                                self.camera_names.append(f)
                                print(f"选择相机文件夹: {f}")

            # for f in os.listdir(folder_path):
            #     if os.path.isdir(os.path.join(folder_path, f)):
            #         for end in self.cameraclass:
            #             if f.endswith(end):
            #                 self.camera_names.append(f)
            #                 print(f"选择相机文件夹: {f}")
        
        if not self.camera_names:
            print("警告: 没有找到符合条件的相机文件夹")
            return
        
        max_sequence_length = 0
        print("正在计算最长序列长度...")
        for folder_name in tqdm.tqdm(folders):
            folder_path = os.path.join(self.input_path, folder_name)
            json_path = os.path.join(folder_path, 'state.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    max_sequence_length = max(max_sequence_length, len(json_data))
        print(f"最长序列长度 【{max_sequence_length}】")
        
        # 使用线程池并行处理文件夹
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(self.process_folder, folder_name, folder_idx, len(folders), max_sequence_length) 
                      for folder_idx, folder_name in enumerate(folders)]
            
            # 使用tqdm显示总体进度
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), 
                                   total=len(futures), 
                                   desc="总体处理进度"):
                try:
                    future.result()
                except Exception as e:
                    print(f"处理文件夹时出错: {e}")
    
    def process_folder(self, folder_name, folder_idx, total_folders, max_sequence_length=0):
        """处理单个文件夹的方法"""
        try:
            folder_path = os.path.join(self.input_path, folder_name)
            
            local_datas = {}
            # 直接使用提前检测好的相机名称
            local_camera_names = self.camera_names
                        
            # 初始化数据字典
            local_datas = {
                '/observations/qpos': [],
                '/action': [],
                '/observations/robot_joint_state': [],
                '/observations/robot_joint_vel': [],
                '/robot_joint_action': [],
            }

            # 存储上一帧的RGB图像,用于计算光流
            prev_frames = {}

            for cam_name in local_camera_names:
                local_datas[f'/observations/images/{cam_name}'] = []
                # 为RGB相机添加光流字典项
                if cam_name.endswith('camera'):
                    prev_frames[cam_name] = None
                    local_datas[f'/observations/images/{cam_name}_flow'] = []
                # 添加字典项
                if cam_name.endswith('mask'):
                    local_datas[f'/observations/images/{cam_name}_attention'] = []
                    local_datas[f'/observations/images/{cam_name}_attention_uni'] = []
        
            # 读取状态文件
            json_path = os.path.join(folder_path, 'state.json')
            if not os.path.exists(json_path):
                print(f"找不到状态文件: {json_path}")
                return
            
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        
            # 检查帧数量是否匹配
            frame_count = len(json_data)
            if not local_camera_names:
                print(f"警告: 文件夹 {folder_name} 没有找到摄像头文件夹,跳过此文件夹")
                return
            
            camera_frame_count = len(glob.glob(os.path.join(folder_path, local_camera_names[0], '*.png')))
        
            if frame_count != camera_frame_count:
                print(f"警告: 文件夹 {folder_name} JSON文件中的帧数 ({frame_count}) 与摄像头文件夹中的帧数 ({camera_frame_count}) 不匹配")
                return
        
            # 处理每一帧数据,添加进度条显示
            frame_ids = sorted(json_data.keys(), key=lambda x: int(x))
            for frame_idx_str in tqdm.tqdm(frame_ids, 
                                        desc=f"文件夹 {folder_name} 进度", 
                                        leave=False):
                jsondata = json_data[frame_idx_str]
                frame_idx = int(frame_idx_str)
                
                # 添加机器人状态和抓取状态
                local_datas['/observations/qpos'].append(jsondata["robot_state"] + [jsondata['grasp_state'][0]])
                local_datas['/action'].append(jsondata['robot_action'] + [jsondata['grasp_action'][0]])
                local_datas['/observations/robot_joint_state'].append(jsondata["robot_joint_state"] + [jsondata['grasp_state'][0]])
                local_datas['/observations/robot_joint_vel'].append(jsondata["robot_joint_vel"])
                local_datas['/robot_joint_action'].append(jsondata["robot_joint_action"] + [jsondata['grasp_action'][0]]) 
                
                # 处理每个摄像头的图像
                for cam_name in local_camera_names:
                    img_path = os.path.join(folder_path, cam_name, f"{frame_idx}.png")
                    if not os.path.exists(img_path):
                        print(f"找不到图像: {img_path}")
                        continue
                    # 读取并处理图像
                    img = Image.open(img_path)

                    # 检查图像尺寸,如果不是指定尺寸就调整
                    if img.size != (self.image_width, self.image_height):
                        img = img.resize((self.image_width, self.image_height))

                    # 转换为numpy数组
                    img_array = np.array(img)

                    if cam_name.endswith('mask'):
                        if self.taskname.startswith("pick_and_lift"):
                            mask_rgb_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
                            # 根据灰度值设置不同的RGB值
                            # mask_rgb_array[(img_array == 35) | (img_array == 31) | (img_array == 34) , 0] = 255
                            # mask_rgb_array[img_array == 84, 1] = 255
                            # mask_rgb_array[img_array == 83, 2] = 255
                            # img_array = np.clip(mask_rgb_array, 0, 255).astype(np.uint8)
                            # 相关物体全部设置为白色
                            target_values = [44, 45, 40, 39, 41, 42, 84, 83, 35, 31, 34]
                            mask = np.isin(img_array, target_values)
                            mask_rgb_array[mask] = 255  # 一次性设置所有匹配像素的所有通道为白色
                            img_array = np.clip(mask_rgb_array, 0, 255).astype(np.uint8)

                        elif self.taskname.startswith("push_button"):
                            mask_rgb_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
                            # 根据灰度值设置不同的RGB值
                            mask_rgb_array[(img_array == 35) | (img_array == 31) | (img_array == 34) , 0] = 255
                            mask_rgb_array[(img_array == 85) | (img_array == 86), 1] = 255
                            mask_rgb_array[(img_array == 81) , 2] = 255
                            img_array = np.clip(mask_rgb_array, 0, 255).astype(np.uint8)
                        elif self.taskname == "pick_and_lift_norot_wzf":
                            # # 保存应该关注的mask，机械臂、物体和终点，对应的真实RGB
                            # mask_real_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
                            # mask_real_array[(img_array == 35) | (img_array == 31) | (img_array == 34)] = 255
                            # mask_real_array[img_array == 84] = 255
                            # mask_real_array[img_array == 83] = 255

                            # 保存应该关注的mask，机械臂、物体和终点，全通道255
                            mask_attention_uni_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
                            mask_attention_uni_array[(img_array == 35) | (img_array == 31) | (img_array == 34)] = 255
                            mask_attention_uni_array[img_array == 84] = 255
                            mask_attention_uni_array[img_array == 83] = 255
                            mask_attention_uni_array = np.clip(mask_attention_uni_array, 0, 255).astype(np.uint8)
                            local_datas[f'/observations/images/{cam_name}_attention_uni'].append(mask_attention_uni_array)
                            # 保存应该关注的mask，机械臂、物体和终点，人为编码
                            mask_attention_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
                            mask_attention_array[(img_array == 35) | (img_array == 31) | (img_array == 34), 0] = 255
                            mask_attention_array[img_array == 84, 1] = 255
                            mask_attention_array[img_array == 83, 2] = 255
                            mask_attention_array = np.clip(mask_attention_array, 0, 255).astype(np.uint8)
                            local_datas[f'/observations/images/{cam_name}_attention'].append(mask_attention_array)
                            # 保存全分割，三通道都是相同的灰度值
                            mask_all_rgb_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
                            mask_all_rgb_array = img_array[..., None]
                            mask_all_rgb_array = np.clip(mask_all_rgb_array, 0, 255).astype(np.uint8)
                            img_array = np.clip(mask_all_rgb_array, 0, 255).astype(np.uint8)


                    # 保存图像到对应的字典
                    local_datas[f'/observations/images/{cam_name}'].append(img_array)
                    
                    # 为RGB相机计算光流
                    if cam_name.endswith('camera'):
                        if prev_frames[cam_name] is None:
                            # 第一帧,自己和自己比较
                            gray1 = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            gray2 = gray1.copy()
                            # 这种情况下光流应该为全零
                        else:
                            # 使用上一帧和当前帧
                            gray1 = cv2.cvtColor(prev_frames[cam_name], cv2.COLOR_RGB2GRAY)
                            gray2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        
                        # 计算光流
                        '''
                        这个函数实现了Farneback光流算法,用于计算两帧之间的密集光流。参数含义如下:
                        gray1:前一帧的灰度图像
                        gray2:当前帧的灰度图像
                        None:输出的光流矩阵,如果为None则函数会自动创建
                        pyr_scale=0.5:图像金字塔的缩放因子,值为0.5表示每一层比上一层小两倍
                        levels=3:金字塔的层数,层数越多可以捕获更大范围的运动
                        winsize=15:平均窗口大小,用于查找局部像素运动,值越大捕获整体运动效果越好
                        iterations=3:在每个金字塔层级上的迭代次数,迭代越多结果越精确但计算量越大
                        poly_n=5:用于像素邻域多项式展开的大小,一般为5或7,表示使用5x5窗口计算多项式系数
                        poly_sigma=1.2:高斯标准差,用于平滑导数计算,值在1.1-1.5之间较为常用
                        flags=0:特殊标志,0表示使用默认参数
                        '''
                        flow = cv2.calcOpticalFlowFarneback(
                            gray1, gray2, None, 
                            pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
                            poly_n=5, poly_sigma=1.2, flags=0
                        )
                        
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        
                        if False: # HSV 映射
                            hsv = np.zeros_like(img_array)
                            hsv[..., 1] = 255
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                            local_datas[f'/observations/images/{cam_name}_flow'].append(flow_rgb)
                        else: # 直接单通道位移图
                            # 直接使用位移幅度(mag)作为像素值，并归一化到0-255范围
                            flow_gray = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            # 由于存储结构要求RGB格式，将灰度图转换为3通道图像
                            flow_rgb = cv2.cvtColor(flow_gray, cv2.COLOR_GRAY2BGR)
                            
                        local_datas[f'/observations/images/{cam_name}_flow'].append(flow_rgb)
                        # 更新上一帧
                        prev_frames[cam_name] = img_array.copy()

        
            # 设置输出文件路径
            record_path = os.path.join(self.output_path, f"episode_{folder_name}.hdf5")
            
            # 保存为HDF5文件
            self.save_to_hdf5(record_path, local_datas, local_camera_names)
            
            print(f"完成处理文件夹 [{folder_idx+1}/{total_folders}]: {folder_name}")
            # 返回处理结果
            return folder_name
        except Exception as e:
            print(f"处理文件夹 {folder_name} 时出错: {str(e)}")
            raise  # 重新抛出异常以便在主线程中捕获
    
    def save_to_hdf5(self, record_path, datas, camera_names):
        # 获取时间步长
        max_timesteps = len(datas['/action'])
        
        with h5py.File(record_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            # 设置全局参数
            root.attrs['sim'] = False
            
            # 创建观测组
            obs = root.create_group('observations')
            
            # 创建图像组，同时包含原始图像和光流图像
            image = obs.create_group('images')
            for cam_name in camera_names:
                if cam_name.endswith('depth'):
                    _ = image.create_dataset(cam_name, (max_timesteps, self.image_height, self.image_width), 
                        dtype='uint8', chunks=(1, self.image_height, self.image_width), 
                        # compression='gzip', compression_opts=2, shuffle=True)
                        )
                elif cam_name.endswith('mask'):
                    _ = image.create_dataset(cam_name, (max_timesteps, self.image_height, self.image_width, 3), 
                        dtype='uint8', chunks=(1, self.image_height, self.image_width, 3), 
                        # compression='gzip', compression_opts=2, shuffle=True)
                        )
                elif cam_name.endswith('camera'):
                    _ = image.create_dataset(cam_name, (max_timesteps, self.image_height, self.image_width, 3), 
                        dtype='uint8', chunks=(1, self.image_height, self.image_width, 3), 
                        # compression='gzip', compression_opts=2, shuffle=True)
                        )
                    # 为RGB相机添加光流数据集
                    _ = image.create_dataset(f"{cam_name}_flow", (max_timesteps, self.image_height, self.image_width, 3), 
                        dtype='uint8', chunks=(1, self.image_height, self.image_width, 3), 
                        # compression='gzip', compression_opts=2, shuffle=True)
                        )

            # 修改维度
            qpos = obs.create_dataset('qpos', (max_timesteps, 8))
            qvel = obs.create_dataset('qvel', (max_timesteps, 8))
            action = root.create_dataset('action', (max_timesteps, 8))
            
            # 创建三个新的数据集
            if '/observations/robot_joint_state' in datas and datas['/observations/robot_joint_state']:
                joint_dim = len(datas['/observations/robot_joint_state'][0])
                robot_joint_state = obs.create_dataset('robot_joint_state', (max_timesteps, joint_dim))
            
            if '/observations/robot_joint_vel' in datas and datas['/observations/robot_joint_vel']:
                joint_vel_dim = len(datas['/observations/robot_joint_vel'][0])
                robot_joint_vel = obs.create_dataset('robot_joint_vel', (max_timesteps, joint_vel_dim))
            
            if '/robot_joint_action' in datas and datas['/robot_joint_action']:
                joint_action_dim = len(datas['/robot_joint_action'][0])
                robot_joint_action = root.create_dataset('robot_joint_action', (max_timesteps, joint_action_dim))
            
            # 存入数据
            for name, array in datas.items():
                if array:  # 只处理非空数据
                    root[name][...] = np.array(array)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'act_plus_plus', 'detr', 'config')),
    config_name="default"
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    data_proccess_config = cfg["data_proccess_config"]

    save_path_head = data_proccess_config['save_path_head']
    save_path_end = data_proccess_config['save_path_end']
    taskname = data_proccess_config['taskname']
    image_width = data_proccess_config['image_width']
    image_height = data_proccess_config['image_height']
    max_workers = data_proccess_config['threads']
    cameraclass = data_proccess_config['cameraclass']

    # 计算路径
    task_path = os.path.join(save_path_head, taskname)
    variation_path = os.path.join(task_path, save_path_end)

    rootpath = pathlib.Path(__file__).parent
    input_path = rootpath.joinpath(variation_path)
    output_path = input_path.parent.joinpath(input_path.name + "_hdf5")

    # 检查输入与输出路径
    if not os.path.exists(input_path):
        print(f"错误:输入路径不存在: {input_path}")
        return
    if os.path.exists(output_path):
        user_input = input("输出路径已经存在,是否覆盖？(y/n): ")
        if user_input.lower() == 'y':
            import shutil
            shutil.rmtree(output_path)
            print(f"已删除现有目录: {output_path}")
        else:
            print("操作已取消")
            return

    output_path.mkdir(parents=True, exist_ok=True)
    print("根目录:", rootpath)
    print("输入路径:", input_path)
    print("输出路径:", output_path)
    converter = RawToHDF5Converter(input_path, output_path,
                                   image_width=image_width,
                                   image_height=image_height,
                                   taskname=taskname,
                                   cameraclass=cameraclass)

    # 如果未指定线程数,使用处理器核心数
    if max_workers is None:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()

    print(f"开始转换,使用 {max_workers} 个线程进行并行处理")
    start_time = time.time()
    converter.convert(max_workers=max_workers)
    end_time = time.time()
    print(f"转换完成,总共耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
