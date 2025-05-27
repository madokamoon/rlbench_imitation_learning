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

# 导入权重计算函数
from weight.weight import calculate_change_weight

OmegaConf.register_new_resolver("eval", eval, replace=True)

class RawToHDF5Converter:
    def __init__(self, input_path, output_path, image_width=640, image_height=480, end_pad=False, weightflag=False):
        self.input_path = input_path
        self.output_path = output_path
        self.camera_names = []
        self.datas = {}
        self.record_path = ""
        self.prev_frames = {}  # 存储前一帧的图像
        self.image_width = image_width
        self.image_height = image_height
        self.end_pad = end_pad 
        self.weightflag = weightflag 
        
    def convert(self, max_workers=None):
        folders = [f for f in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, f))]
        if not folders:
            print("输入路径下没有文件夹")
            return
        
        print(f"找到以下文件夹: {folders}")
        print(f"共计 {len(folders)} 个文件夹需要处理")
        
        # 当 end_pad=True 时，预先计算最长序列长度
        max_sequence_length = 0
        if self.end_pad:
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
                    # 获取处理结果（如果有异常会在这里抛出）
                    future.result()
                except Exception as e:
                    print(f"处理文件夹时出错: {e}")
    
    def process_folder(self, folder_name, folder_idx, total_folders, max_sequence_length=0):
        """处理单个文件夹的方法"""
        try:
            folder_path = os.path.join(self.input_path, folder_name)
            
            # 为每个线程创建独立的数据结构
            local_datas = {}
            local_camera_names = []
            local_prev_frames = {}

            for f in os.listdir(folder_path):
                if os.path.isdir(os.path.join(folder_path, f)):
                    if f.endswith('depth'):
                        # pass # 不转换深度图像
                        local_camera_names.append(f)
                    elif f.endswith('mask'):
                        # pass # 不转换mask图像
                        local_camera_names.append(f)
                    elif f.endswith('camera'):
                        local_camera_names.append(f)
                        
            # 初始化数据字典
            local_datas = {
                '/observations/qpos': [],
                '/action': [],
                # 添加三个新的数据项
                '/observations/robot_joint_state': [],
                '/observations/robot_joint_vel': [],
                '/robot_joint_action': [],
            }
            
            # 为每个摄像头创建数据项和权重项
            for cam_name in local_camera_names:
                local_datas[f'/observations/images/{cam_name}'] = []
                local_datas[f'/observations/weight/{cam_name}'] = []  # 初始化权重列表
            
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
                print(f"警告: 文件夹 {folder_name} 没有找到摄像头文件夹，跳过此文件夹")
                return
                
            camera_frame_count = len(glob.glob(os.path.join(folder_path, local_camera_names[0], '*.png')))
            
            if frame_count != camera_frame_count:
                print(f"警告: 文件夹 {folder_name} JSON文件中的帧数 ({frame_count}) 与摄像头文件夹中的帧数 ({camera_frame_count}) 不匹配")
                return
            
            # 处理每一帧数据，添加进度条显示
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
                
                # 临时存储该帧所有摄像头的原始权重
                frame_weights = {}
                
                # 处理每个摄像头的图像
                for cam_name in local_camera_names:
                    img_path = os.path.join(folder_path, cam_name, f"{frame_idx}.png")
                    if not os.path.exists(img_path):
                        # print(f"找不到图像: {img_path}")
                        continue
                    
                    # 读取并处理图像
                    img = Image.open(img_path)
                    
                    # 检查图像尺寸，如果不是指定尺寸就调整
                    if img.size != (self.image_width, self.image_height):
                        img = img.resize((self.image_width, self.image_height))

                    # 转换为numpy数组
                    img_array = np.array(img)
                    local_datas[f'/observations/images/{cam_name}'].append(img_array)
                    
                    # 计算权重（与前一帧比较）
                    if self.weightflag:
                        if frame_idx > 0 and cam_name in local_prev_frames:
                            # 计算当前帧与前一帧的变化权重
                            weight = calculate_change_weight(local_prev_frames[cam_name], img_array)
                        else:
                            # 第一帧或无前一帧数据时，设置默认权重1.0
                            weight = 1.0
                    else:
                        # 如果weightflag为False，直接设置权重为1.0
                        weight = 1.0
                    
                    # 暂存权重
                    frame_weights[cam_name] = weight
                    
                    # 更新前一帧图像
                    local_prev_frames[cam_name] = img_array.copy()
                
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
                                local_datas[f'/observations/weight/{cam_name}'].append(norm_weight)
                        else:
                            # 如果总权重为零，所有mask摄像头权重设为1
                            for cam_name in mask_cams:
                                local_datas[f'/observations/weight/{cam_name}'].append(1.0)

                    # 处理camera摄像头的权重 - 全部设置为1
                    for cam_name in camera_cams:
                        local_datas[f'/observations/weight/{cam_name}'].append(1.0)
            
            # 如果启用了 end_pad，并且当前序列长度小于最大序列长度，则进行填充
            current_length = len(local_datas['/action'])
            if self.end_pad and current_length < max_sequence_length and current_length > 0:
                padding_length = max_sequence_length - current_length
                print(f"文件夹 {folder_name} 需要填充 {padding_length} 帧")
                
                # 填充各个数据项
                for key, data_list in local_datas.items():
                    if not data_list:  # 跳过空列表
                        continue
                    
                    # 获取最后一帧的数据
                    last_frame = data_list[-1]
                    
                    # 对于图像类型数据 (numpy数组)
                    if isinstance(last_frame, np.ndarray) and last_frame.ndim > 1:
                        # 重复最后一帧 padding_length 次
                        padding = [last_frame.copy() for _ in range(padding_length)]
                    else:  # 对于其他类型数据 (关节状态、动作等)
                        # 重复最后一帧 padding_length 次
                        padding = [last_frame for _ in range(padding_length)]
                    
                    # 添加填充到数据列表
                    local_datas[key].extend(padding)
            
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
            
            # 创建图像组
            image = obs.create_group('images')
            for cam_name in camera_names:
                if cam_name.endswith('depth'):
                    _ = image.create_dataset(cam_name, (max_timesteps, self.image_height, self.image_width), 
                        dtype='uint8', chunks=(1, self.image_height, self.image_width)) 
                elif cam_name.endswith('mask'):
                    _ = image.create_dataset(cam_name, (max_timesteps, self.image_height, self.image_width, 3), 
                        dtype='uint8', chunks=(1, self.image_height, self.image_width, 3)) 
                elif cam_name.endswith('camera'):
                    _ = image.create_dataset(cam_name, (max_timesteps, self.image_height, self.image_width, 3), 
                        dtype='uint8', chunks=(1, self.image_height, self.image_width, 3)) 


            # 创建权重组
            weight = obs.create_group('weight')
            
            # 创建权重数据集
            for cam_name in camera_names:
                _ = weight.create_dataset(cam_name, (max_timesteps,), dtype='float32')
        
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
    end_pad = data_proccess_config['end_pad']
    max_workers = data_proccess_config['threads']
    weightflag = data_proccess_config['weightflag']

    # 保存路径
    task_path = os.path.join(save_path_head, taskname)
    if save_path_end == "":
        now_time = datetime.datetime.now()
        str_time = now_time.strftime("%Y-%m-%d-%H-%M-%S")
        variation_path = os.path.join(task_path, str_time)
        save_path_end = str_time
    else:
        variation_path = os.path.join(task_path, save_path_end)

    rootpath = pathlib.Path(__file__).parent
    input_path = rootpath.joinpath(variation_path)
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
        else:
            print("操作已取消")
            return

    output_path.mkdir(parents=True, exist_ok=True)
    print("----------开始转换---------")
    print("根目录：", rootpath)
    print("输入路径：", input_path)
    print("输出路径：", output_path)
    converter = RawToHDF5Converter(input_path, output_path,
                                   image_width=image_width,
                                   image_height=image_height,
                                   end_pad=end_pad,
                                   weightflag=weightflag)

    # 如果未指定线程数，使用处理器核心数
    if max_workers is None:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()

    print(f"使用 {max_workers} 个线程进行并行处理")
    start_time = time.time()
    converter.convert(max_workers=max_workers)
    end_time = time.time()
    print("----------转换完成----------")
    print(f"总共耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
