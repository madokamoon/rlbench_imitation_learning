import pathlib
import json
import copy
import numpy as np
import yaml
import imageio.v2 as imageio
import concurrent.futures
import tqdm
from replay_buffer import ReplayBuffer
import os
import h5py
from PIL import Image
import glob


class RawToHDF5Converter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.camera_names = []
        self.datas = {}
        self.record_path = ""
        
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
            self.camera_names = []
            self.datas = {}
            
            # 修改检测摄像头文件夹的代码，排除depth和mask结尾的文件夹
            self.camera_names = [f for f in os.listdir(folder_path) 
                               if os.path.isdir(os.path.join(folder_path, f)) 
                               and not f.endswith('depth') 
                               and not f.endswith('mask')]
            print(f"找到以下摄像头: {self.camera_names}")
            print(f"摄像头数量: {len(self.camera_names)}")
            
            # 初始化数据字典
            self.datas = {
                '/observations/qpos': [],
                # '/observations/qvel': [],
                '/action': [],
            }
            
            # 为每个摄像头创建数据项
            for cam_name in self.camera_names:
                self.datas[f'/observations/images/{cam_name}'] = []
            
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
                print("警告: 没有找到摄像头文件y夹，跳过此文件夹")
                continue
                
            camera_frame_count = len(glob.glob(os.path.join(folder_path, self.camera_names[0], '*.png')))
            
            if frame_count != camera_frame_count:
                print(f"警告: JSON文件中的帧数 ({frame_count}) 与摄像头文件夹中的帧数 ({camera_frame_count}) 不匹配")
                continue
            
            print(f"处理 {frame_count} 帧数据...")
            
            # 处理每一帧数据
            for frame_idx_str in tqdm.tqdm(sorted(json_data.keys(), key=lambda x: int(x)), 
                                     desc=f"处理文件夹 {folder_name}", 
                                     total=frame_count):
                jsondata = json_data[frame_idx_str]
                frame_idx = int(frame_idx_str)
                
                # 添加机器人状态和抓取状态
                self.datas['/observations/qpos'].append(jsondata["robot_state"] + [jsondata['grasp_state'][0]])
                # self.datas['/observations/qvel'].append(jsondata['robot_vel_command'] + [jsondata['grasp_action'][0]])
                self.datas['/action'].append(jsondata['robot_action'] + [jsondata['grasp_action'][0]])
                
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
                        # print(f"调整图像尺寸从 {img.size} 到 (640, 480)",end='\r')
                        img = img.resize((640, 480))
                    
                    # 确保图像是RGB格式
                    # if img.mode != 'RGB':
                        # print(f"图像模式不正确，转换为RGB: {img.mode}",end='\r')
                        # img = img.convert('RGB')

                    # 转换为numpy数组
                    img_array = np.array(img)
                    self.datas[f'/observations/images/{cam_name}'].append(img_array)
            
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
            for cam_name in self.camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), 
                                        dtype='uint8', chunks=(1, 480, 640, 3))
            
            # 修改维度
            qpos = obs.create_dataset('qpos', (max_timesteps, 8))
            qvel = obs.create_dataset('qvel', (max_timesteps, 8))
            action = root.create_dataset('action', (max_timesteps, 8))
            
            # 存入数据
            for name, array in self.datas.items():
                root[name][...] = np.array(array)



class Tools:
    def __init__(self):
        # 读取配置文件
        rootpath = pathlib.Path(__file__).parent
        file_yaml = rootpath.joinpath('config_tools.yaml') 
        rf = open(file=file_yaml, mode='r', encoding='utf-8')
        crf = rf.read()
        rf.close()  # 关闭文件
        yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)
        # 读取使用的工具和相应的配置
        tool_choices = yaml_data["tool_choice"]
        self.tool_choice = None
        for tool_choice in tool_choices:
            if tool_choice[0] == "*":
                self.tool_choice = tool_choice[1:]
        assert self.tool_choice is not None
        print("Tool name: {}".format(self.tool_choice))
        self.tool_config = yaml_data[self.tool_choice]

    def run(self):
        assert hasattr(self, self.tool_choice)
        getattr(self, self.tool_choice)()

    def formatconvert_pngs2mp4(self):
        """
        作用：
            很多张png图片转mp4
        Args:

        Returns:

        """
        # 为多线程服务的转换函数
        def convert_pictures_to_mp4(cameras_folder, cameras_episode_output_path, mp4_fps):
            camera_idx_str = cameras_folder.name.replace("camera", "")
            png_files = sorted(cameras_folder.glob("*.png"), key=lambda x: int(x.stem))
            assert len(png_files) > 0
            mp4_output_path = pathlib.Path(cameras_episode_output_path, f"{camera_idx_str}.mp4")
            # 读取第一张图片以获取尺寸
            first_image = imageio.imread(png_files[0])
            height, width, _ = first_image.shape
            # 创建视频写入器
            with imageio.get_writer(mp4_output_path, fps=mp4_fps) as writer:
                for png_file in png_files:
                    image = imageio.imread(png_file)
                    writer.append_data(image)

        # 为多线程服务的episode处理函数
        def process_episode(one_episode_files, cameras_output_path, mp4_fps, lock, pbar):
            cameras_folders = list(one_episode_files.glob("camera*"))
            cameras_episode_output_path = pathlib.Path(cameras_output_path, one_episode_files.name)
            cameras_episode_output_path.mkdir(parents=True, exist_ok=True)
            # 使用多线程读取每个相机的数据，并转化成mp4
            for cameras_folder in cameras_folders:
                convert_pictures_to_mp4(cameras_folder, cameras_episode_output_path, mp4_fps)

            # 更新进度条
            with lock:
                pbar.update(1)

        # 读取输入输出路径
        src_path = self.tool_config["src_path"]
        output_path = self.tool_config["output_path"]
        mp4_fps = self.tool_config["mp4_fps"]
        max_workers = self.tool_config["max_workers"]

        # 检查输入路径是否存在
        if not os.path.exists(src_path):
            print(f"错误：输入路径不存在: {src_path}")
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

        print("Src path: {}".format(src_path))
        print("Output path: {}".format(output_path))

        # 创建相机数据输出路径
        cameras_output_path = pathlib.Path(output_path, 'videos')
        cameras_output_path.mkdir(parents=True, exist_ok=True)

        # 获取原始数据，获取json数据，将照片转到mp4
        src = pathlib.Path(src_path)
        sorted_src_files = sorted(list(src.glob("*")), key=lambda x: int(x.stem))
        with tqdm.tqdm(total=len(sorted_src_files), desc=f"Converting pictures to mp4", leave=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                lock = tqdm.tqdm.get_lock()
                futures = [
                    executor.submit(process_episode, one_episode_files, cameras_output_path, mp4_fps, lock, pbar)
                    for one_episode_files in sorted_src_files]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # 捕获异常
                    except Exception as e:
                        print(f"Error processing episode: {e}")

    def formatconvert_raw2dp(self):
        """
        作用：
            将原始数据转化为DP训练所需数据
            pngs->mp4
            json->zarr:
                timestamp=timestamp
                robot_eef_pose(7)=[robot_state(6), grasp_state(1)]
                action(7)=[robot_action(6), grasp_action(1)]
        Args:

        Returns:

        """

        # 为多线程服务的pngs->mp4转换函数
        def convert_pictures_to_mp4(cameras_folder, cameras_episode_output_path, mp4_fps):
            camera_idx_str = cameras_folder.name.replace("camera", "")
            png_files = sorted(cameras_folder.glob("*.png"), key=lambda x: int(x.stem))
            assert len(png_files) > 0
            mp4_output_path = pathlib.Path(cameras_episode_output_path, f"{camera_idx_str}.mp4")
            # 读取第一张图片以获取尺寸
            first_image = imageio.imread(png_files[0])
            height, width, _ = first_image.shape
            # 创建视频写入器
            with imageio.get_writer(mp4_output_path, fps=mp4_fps) as writer:
                for png_file in png_files:
                    image = imageio.imread(png_file)
                    writer.append_data(image)

                # 为多线程服务的episode处理函数

        def process_episode(one_episode_files, cameras_output_path, mp4_fps, replay_buffer, show_output_data, lock,
                            pbar):
            cameras_folders = list(one_episode_files.glob("camera*"))
            cameras_episode_output_path = pathlib.Path(cameras_output_path, one_episode_files.name)
            cameras_episode_output_path.mkdir(parents=True, exist_ok=True)
            for cameras_folder in cameras_folders:
                convert_pictures_to_mp4(cameras_folder, cameras_episode_output_path, mp4_fps)

            # 读取json数据
            raw_json_datas = {}
            dp_zarr_datas = {}
            with open(pathlib.Path(one_episode_files, 'state.json'), 'r', encoding='utf-8') as file:
                # 读取每一个点位数据
                one_episode_json_datas = json.load(file)
                for index, data in one_episode_json_datas.items():
                    for k, v in data.items():
                        if k not in raw_json_datas.keys():
                            raw_json_datas[k] = []
                        raw_json_datas[k].append(v)

            # 将json数据转化成zarr数据
            for k, v in raw_json_datas.items():
                raw_json_datas[k] = np.array(raw_json_datas[k])
            dp_zarr_datas["timestamp"] = raw_json_datas["timestamp"]
            dp_zarr_datas["robot_eef_pose"] = np.concatenate(
                (raw_json_datas["robot_state"], raw_json_datas["grasp_state"]), axis=1)
            dp_zarr_datas["action"] = np.concatenate(
                (raw_json_datas["robot_action"], raw_json_datas["grasp_action"]), axis=1)
            if show_output_data:
                for k, v in dp_zarr_datas.items():
                    print(f"{k}: {v[0]}, shape: {v.shape}")
            replay_buffer.add_episode(dp_zarr_datas, compressors='disk')
            raw_json_datas.clear()
            dp_zarr_datas.clear()

            # 更新进度条
            with lock:
                pbar.update(1)

            # 读取输入输出路径

        src_path = self.tool_config["src_path"]
        output_path = self.tool_config["output_path"]
        mp4_fps = self.tool_config["mp4_fps"]
        show_output_data = self.tool_config["show_output_data"]

        # 检查输入路径是否存在
        if not os.path.exists(src_path):
            print(f"错误：输入路径不存在: {src_path}")
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

        print("Src path: {}".format(src_path))
        print("Output path: {}".format(output_path))

        # 创建相机数据输出路径和replay_buffer并初始化
        zarr_output_path = pathlib.Path(output_path, 'replay_buffer.zarr')
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_output_path, mode='a')
        cameras_output_path = pathlib.Path(output_path, 'videos')
        cameras_output_path.mkdir(parents=True, exist_ok=True)

        # 获取原始数据，获取json数据，将照片转到mp4
        src = pathlib.Path(src_path)
        sorted_src_files = sorted(list(src.glob("*")), key=lambda x: int(x.stem))
        with tqdm.tqdm(total=len(sorted_src_files), desc=f"Converting raw data to dp format", leave=False) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                lock = tqdm.tqdm.get_lock()
                futures = [
                    executor.submit(process_episode, one_episode_files, cameras_output_path, mp4_fps, replay_buffer,
                                    show_output_data, lock, pbar)
                    for one_episode_files in sorted_src_files]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # 捕获异常
                    except Exception as e:
                        print(f"Error processing episode: {e}")

    def formatconvert_raw2act(self):
        """
        作用：
            将原始数据转化为ACT训练所需数据
            pngs->hdf5
            json->hdf5

            self.datas = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
                '/observations/images/{cam_name}': [],
            }

        """
        rootpath = pathlib.Path(__file__).parent.parent
        print(rootpath)
        input_path = rootpath.joinpath(self.tool_config["src_path"]) 
        output_path = rootpath.joinpath(self.tool_config["output_path"])
        
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


def main():
    tools = Tools()
    tools.run()

if __name__ == '__main__':
    main()
