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
```
conda activate rlact
```
使用pip安装到`site-packages`，并非安装本项目的RLbench文件夹

本项目的RLbench文件夹无用，仅方便参考
```
pip install git+https://github.com/stepjam/RLBench.git
```
测试
```
python RLBench/tools/task_builder.py
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

## 文件说明

data_sampler.yaml：集成采集，重现，转换，训练，测试的默认配置

**配置文件优先级：** 命令行指定 > data_sampler_local.yaml > data_sampler.yaml

data_sampler.py：集成数据采集（mode=0），数据重现（mode=1），仿真测试（mode=2）的功能

data_proccess.py ：数据转换功能，转换为hdf5同时计算权重

act_training.py ：训练

act_policy_wrapper.py ：被 data_sampler.py 的 （mode=2） 模式调用




## mode=0 数据收集

运行配置会保存至 `save_path_end/data_sampler_config.yaml`

配置文件中的`taskname` 的可用任务名称参见 `RLBench/rlbench/tasks` 文件夹

```bash
python data_sampler.py
python data_sampler.py --config config.yaml
```
保存路径为 `save_path_head/taskname/save_path_end(空白为时间戳)`

## mode=1 数据重现

利用收集的数据在rlbench中重现动作，如果不是静态模式，仅重现动作

如果是静态模式 `static_positions: True` ，会加载 initial_state.pickle 环境

```bash
python data_sampler.py
python data_sampler.py --config config.yaml
```
## data_proccess.py 数据转换

默认为全部线程

```bash
python data_proccess.py
python data_proccess.py --config config.yaml --threads 12 
```
保存路径为 `save_path_head/taskname/save_path_end(空白为时间戳)_hdf5`

## 可视化 

使用act自带的功能，支持任意维度和任意数量相机

```bash
python act_plus/act_plus_plus/visualize_episodes.py --dataset_dir /home/madoka/python/rlbench_imitation_learning/data/pick_and_lift/30static_hdf5 --episode 0
```

MP4播放工具：`sudo apt-get install smplayer`

hdf5网页工具  https://myhdf5.hdfgroup.org/ 

使用 tools，但tools只支持六维机械臂数据和三个相机

## imitate_episodes.py 训练 

**配置文件方式：**

运行配置会保存至 `ckpt_dir/training_config.yaml`

```bash
python act_training.py
python act_training.py --config config.yaml
```

**原始方式：**

配置依赖 `命令行参数` 和 `data_sampler.yaml 中 的 ['act_policy']['task_config']`

```bash
CUDA_VISIBLE_DEVICES=0 python act_plus_plus/imitate_episodes.py --task_name pick_and_lift --ckpt_dir training/pick_and_lift/20demos_hdf5_4_4000_fwo  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 4 --dim_feedforward 3200 --lr 1e-5 --seed 0 --num_steps 4000
```

## mode=2 仿真测试

```bash
python data_sampler.py
python data_sampler.py --config config.yaml
```


# 三、其他

搜索 `act修改` 可以查看对 act_plus_plus 文件夹内的代码的所有改动

