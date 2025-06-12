# rlbench_imitation_learning

使用 RLBench 收集模仿学习数据 

参考资料

[GitHub - RLBench](https://github.com/stepjam/RLBench)

[CoppeliaSim Doc](https://manual.coppeliarobotics.com/index.html)

[GitHub - Boxjod/RLBench_ACT](https://github.com/Boxjod/RLBench_ACT)

# 一、环境配置

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

启用一个IL算法conda环境，无python版本要求，这里启用下面创建的 rlact
```bash
conda activate rlact
cd RLBench
pip install -e .

# 不需要 pip install git+https://github.com/stepjam/RLBench.git

```
测试
```bash
python RLBench/tools/task_builder.py
```

## 使用虚拟显示器运行 CoppeliaSim

注意：使用--use-display-device=None参数会告诉NVIDIA驱动忽略任何物理连接的显示设备，而只使用虚拟显示器

```bash
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
echo -e 'Section "ServerFlags"\n\tOption "MaxClients" "2048"\nEndSection\n' \
    | sudo tee /etc/X11/xorg.conf.d/99-maxclients.conf
```


然后，当你想要运行 RLBench 时，启动 X
```bash
# nohup and disown is important for the X server to keep running in the background
sudo nohup X :99 & disown
```

使用 glxgears 测试你的显示是否正常工作

如果你有多个 GPU，你可以通过以下方式选择你的 GPU 测试
```bash
DISPLAY=:99 glxgears
DISPLAY=:99.<gpu_id> glxgears
DISPLAY=:99.3 glxgears
```


在运行python的时候，首先要指定显示器
```bash
export DISPLAY=:99
python data_sampler.py
```


显示/关闭虚拟显示器进程
```bash
ps aux | grep "X :99"
sudo kill [PID号]
sudo pkill -f "X :99"
```

恢复原始X配置,连接物理显示器
```bash
sudo rm /etc/X11/xorg.conf
```

## 安装act-plus

参考：[GitHub - MarkFzp/act-plus-plus:](https://github.com/MarkFzp/act-plus-plus.git)
```shell

conda create -n rlact python=3.8.10 
conda activate rlact

pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython

pip install typeguard pyyaml
pip install wandb

cd act-plus-plus/detr/
pip install -e .


# 安装以下内容以运行act-plus中的 Diffusion Policy 但是安装后 numpy 版本冲突 暂不安装
# git clone https://githubfast.com/ARISE-Initiative/robomimic --recurse-submodules
# git checkout r2d2
# pip install -e .
```

##  其他安装

```bash
pip install gymnasium
pip install omegaconf
# 解决rlbench和cv2的qt冲突
pip uninstall opencv-python
pip install opencv-python-headless
# data_proccess需要
pip install imageio
# 配置参数框架
pip install hydra-core==1.2.0
pip install omegaconf
# sam
pip install git+https://github.com/facebookresearch/segment-anything.git  
# depth_anything 别下载到本项目中
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
pip install xformers
```

## 错误解决

### failed to load driver: swrast

执行 python data_sampler.py 如果报错
```bash
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: 无法打开共享目标文件: 没有那个文件或目录 (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
```

查找 /usr/lib/x86_64-linux-gnu/dri 下是否有该缺失的文件
```bash
cd /usr/lib/x86_64-linux-gnu/dri
ls
```

如果有，在 /usr/lib/dri/ 中建立软链接
```bash
cd /usr/lib
sudo mkdir ./dri
sudo ln -s /usr/lib/x86_64-linux-gnu/dri/iris_dri.so /usr/lib/dri/
sudo ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/
```

接下来会报错
```bash
libGL error: MESA-LOADER: failed to open swrast: /home/haoyue/anaconda3/envs/rlact/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1) (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
```
查看 GLIBCXX 当前版本

`which python` → /home/haoyue/anaconda3/envs/rlact/bin/python

`strings /home/haoyue/anaconda3/envs/rlact/lib/libstdc++.so.6 | grep GLIBCXX`  → 发现没有 GLIBCXX_3.4.30

`conda list -n rlact | grep libstdcxx-ng`  → 发现为 11.2.0 应升级到 12.1.0

```bash
conda install libstdcxx-ng=12.1.0 --channel conda-forge
```

再次查看已经有了 GLIBCXX_3.4.30

### rlbench 与 cv2 同时使用导致 Could not find the Qt platform plugin "xcb"

```bash
qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "/home/madoka/APP/anaconda3/envs/rlact/lib/python3.8/site-packages/cv2/qt/plugins"
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
```

只要在rlbench运行，就不能 import cv2 否则qt会去cv2中寻找plugin，而不去 coppeliaSim 中寻找

解决：使用cv2无头模式

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

需要时暂时使用有头模式：

```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

### Coppeliasim GUI 中保存导致 Handle cam_head_mask does not exist 
**不要在 Coppeliasim 的 GUI 中保存代码** ，无论是使用 Ctrl+S 还是在关闭窗口时在弹出窗口中确认 “Did you save changes？”在 GUI 中保存场景可能会导致缺少组件，并导致后续执行中出现错误。例如：

`"RuntimeError: Handle cam_head_mask does not exist"`

如果不小心保存了更改，请使用本项目的`task_design.ttt`替换conda环境中的`task_design.ttt `

例如我的环境为 `/home/madoka/APP/anaconda3/envs/rlact/lib/python3.8/site-packages`

项目路径为 `/home/madoka/python/rlbench_imitation_learning`

则运行以下命令：

```bash
cd /home/madoka/APP/anaconda3/envs/rlact/lib/python3.8/site-packages/rlbench
rm task_design.ttt 
cp /home/madoka/python/rlbench_imitation_learning/RLBench/rlbench/task_design.ttt task_design.ttt 
```


# 二、使用说明

## 文件

```bash
rlbench_imitation_learning
├── act_plus_plus           # act源码
├── RLBench                 # Rlbench源码
├── mytasks                 # 自定义Rlbench任务
├── config_symlink -> act_plus_plus/detr/config     # 配置文件软连接
├── test_scripts            # 测试脚本
├── tools                   # 工具
├── weight                  # 权重计算
├── backfile                # 备份和弃用文件
├── data_sampler.py         # 数据收集 重现
├── data_proccess.py        # 数据处理
├── act_training.py         # 训练
├── act_eval.py             # 测试
├── act_policy_wrapper.py   # 测试用
├── bash_train_and_eval.sh  # 批量训练测试
├── eval_results.csv        # 测试结果          
├── sweeps.yaml             # wandb参数扫描
├── foundation_ckpt         # 基础模型权重(ignore)
├── data                    # 演示数据(ignore)
├── outputs                 # hydra输出(ignore)
├── training                # 训练结果(ignore)
├── wandb                   # wandb输出(ignore)
└── README.md   
```

**配置文件优先级：** 命令行指定 `--config-name=***.yaml` > 默认配置 `default.yaml`


## 数据收集

运行配置会保存至 `save_path_end/data_sampler_config.yaml`

配置文件中的`taskname` 的可用任务名称参见 `RLBench/rlbench/tasks` 文件夹

```bash
python data_sampler.py
```
保存路径为 `save_path_head/taskname/save_path_end(空白为时间戳)`

## 数据重现

利用收集的数据在rlbench中重现动作，如果不是静态模式，仅重现动作

如果是静态模式 `static_positions: True` ，会加载 initial_state.pickle 环境

```bash
python data_sampler.py
```
## 数据转换

默认为全部线程

```bash
python data_proccess.py
```
保存路径为 `save_path_head/taskname/save_path_end(空白为时间戳)_hdf5`

## 可视化hdf5

使用act自带的功能，支持任意维度和任意数量相机

```bash
python act_plus_plus/visualize_episodes.py --dataset_dir data/push_button/100demos_hdf5 --episode 0
```

MP4播放工具：`sudo apt-get install smplayer`

hdf5网页工具  https://myhdf5.hdfgroup.org/ 

使用tools，但tools只支持六维机械臂数据和三个相机

## 训练 

运行配置会保存至 `ckpt_dir/training_config.yaml`

```bash
python act_training.py --config-name=***.yaml
```

## 仿真测试

使用 `--ckpt` 指定需要测试的训练结果

```bash
python act_eval.py --ckpt training/act_policy_pick_and_lift/100demos

# 其他参数：
--episode_len, default=250
--ckpt_name, default='policy_last.ckpt'
--show_3D_state, default=False
--show_transform_attention, default=False
--temporal_agg, default=False
```

## 参数扫描

```bash
wandb sweep --project projectname sweeps.yaml
wandb agent --count 5 user/projectname/runid
```

# 三、其他

## 修改

搜索 `act修改` 可以查看对 act_plus_plus 文件夹内的代码的所有改动

搜索 `rlbench修改` 可以查看对 RlBench 文件夹内的代码的所有改动

## backfile 备份和弃用的文件

`data_proccess_back.py` 6.9 23:22 新文件删除了weight和pad功能,加入光流
