# rlbench_imitation_learning

使用 RLBench 收集模仿学习数据 

参考资料
[GitHub - RLBench](https://github.com/stepjam/RLBench)

[CoppeliaSim Doc](https://manual.coppeliarobotics.com/index.html)

[GitHub - Boxjod/RLBench_ACT](https://github.com/Boxjod/RLBench_ACT)

# 环境配置

## 安装CoppeliaSim

```bash
# ~/.bashrc 加入 CoppeliaSim 安装位置
export COPPELIASIM_ROOT=${HOME}/install/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export PATH=$COPPELIASIM_ROOT:$PATH

# 安装 CoppeliaSim
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```
测试
```
coppeliaSim.sh
```
## 安装RLBench

启用一个IL算法conda环境，无python版本要求
```
conda activate rlact
```
使用pip安装到`site-packages`，并非安装本项目的RLbench文件夹
```
pip install git+https://github.com/stepjam/RLBench.git
```
测试
```
python RLBench/tools/task_builder.py
```


# 收集数据

## 不要在 GUI 中保存
**不要在 Coppeliasim 的 GUI 中保存代码** ，无论是使用 Ctrl+S 还是在关闭窗口时在弹出窗口中确认 “Did you save changes？”在 GUI 中保存场景可能会导致缺少组件，并导致后续执行中出现错误。例如：

`"RuntimeError: Handle cam_head_mask does not exist"`

如果不小心保存了更改，请使用本项目的`task_design.ttt`替换conda环境中的`task_design.ttt `
```bash
# 例如我的环境为 /home/madoka/APP/anaconda3/envs/rlact/lib/python3.8/site-packages
# 项目路径为 /home/madoka/python/rlbench_imitation_learning
cd /home/madoka/APP/anaconda3/envs/rlact/lib/python3.8/site-packages/rlbench
rm task_design.ttt 
cp /home/madoka/python/rlbench_imitation_learning/RLBench/rlbench/task_design.ttt task_design.ttt 
```

## 选择任务

参见 `RLBench/rlbench/tasks` 文件夹

例如：选择 `put_rubbish_in_bin.py` 查看其中任务类名称为 `PutRubbishInBin`

填入`data_sampler_hdf5.yaml`中的`taskclassname`

## 选择观测参数

`data_sampler.yaml`支持配置
- 图像大小
- 启用的相机

逐个demo保存并释放内存，能支持一个demo的内存即可

更多ObservationConfig 可配置参数：

| 属性名                       | 类型             | 说明                  |
| ------------------------- | -------------- | ------------------- |
| `front_camera`            | `CameraConfig` | 前置摄像头配置（RGB/深度/分割等） |
| `left_shoulder_camera`    | `CameraConfig` | 左肩摄像头配置             |
| `right_shoulder_camera`   | `CameraConfig` | 右肩摄像头配置             |
| `overhead_camera`         | `CameraConfig` | 俯视摄像头配置             |
| `wrist_camera`            | `CameraConfig` | 手腕摄像头配置             |
| `wrist_camera_matrix`     | `bool`         | 是否返回手腕摄像头的 4x4 变换矩阵 |
| `gripper_open`            | `bool`         | 返回夹爪是否张开（开/合）       |
| `gripper_pose`            | `bool`         | 返回夹爪的世界位姿（位置 + 方向）  |
| `gripper_joint_positions` | `bool`         | 返回夹爪的各个关节角度         |
| `gripper_matrix`          | `bool`         | 返回夹爪的 4x4 变换矩阵      |
| `gripper_touch_forces`    | `bool`         | 返回夹爪的触觉接触力量         |
| `joint_positions`         | `bool`         | 返回机械臂的各个关节位置        |
| `joint_velocities`        | `bool`         | 返回机械臂的各个关节速度        |
| `joint_forces`            | `bool`         | 返回机械臂的各个关节受力（力矩）    |
| `joint_positions_noise`   | `NoiseModel`   | 对关节位置加入噪声的模型        |
| `joint_velocities_noise`  | `NoiseModel`   | 对关节速度加入噪声的模型        |
| `joint_forces_noise`      | `NoiseModel`   | 对关节力加入噪声的模型         |
| `task_low_dim_state`      | `bool`         | 是否返回任务的低维状态（手工特征）   |
| `record_gripper_closing`  | `bool`         | 是否记录夹爪闭合过程（用于数据集生成） |


## 收集数据

```bash
python data_sampler.py
```
## 可视化
```bash
python act_plus/act_plus_plus/visualize_episodes.py --dataset_dir /home/madoka/python/rlbench_imitation_learning/data/pick_and_lift/30static_hdf5 --episode 0
```


## 问题

rlbench 不能与 cv2 同时使用，否则会报错：

```bash
qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "/home/madoka/APP/anaconda3/envs/rlact/lib/python3.8/site-packages/cv2/qt/plugins"
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
```

解决

```
pip uninstall opencv-python
pip install opencv-python-headless

pip uninstall opencv-python-headless
pip install opencv-python
```

# ACT

[ACTPLUS.md](act-plus-plus/ACTPLUS.md)

