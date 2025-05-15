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




# 二、使用说明

## 不要在 GUI 中保存
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

## 文件说明

data_sampler.yaml：集成采集，重现，转换，训练，测试的配置

data_sampler.py：集成数据采集（mode=0），数据重现（mode=1），仿真测试（mode=2）的功能

> 数据重现指利用收集的数据在rlbench中重现动作，暂未保存物体初始摆放位置，仅重现动作

data_proccess.py ：数据转换功能，转换为hdf5同时计算权重

act_policy_wrapper.py ：被 data_sampler.py 的 （mode=2） 模式调用

weight.py ：互信息计算

weight_visualization.py ：权重可视化


## mode=0 数据收集

配置文件中的`taskname` 的可用任务名称参见 `RLBench/rlbench/tasks` 文件夹

目前发现 pick_and_lift 和实际任务最相似

```bash
python data_sampler.py
```
保存路径为 `save_path_head + taskname + save_path_end/空白为时间戳`

## data_proccess.py 数据重现
```bash
python data_proccess.py
```
## mode=1 数据转换
```bash
python data_sampler.py
```
保存路径为 `save_path_head + taskname + save_path_end/空白为时间戳_hdf5`

## 可视化 

使用act自带的功能，支持任意维度和任意数量相机

```bash
python act_plus/act_plus_plus/visualize_episodes.py --dataset_dir /home/madoka/python/rlbench_imitation_learning/data/pick_and_lift/30static_hdf5 --episode 0
```

使用 https://myhdf5.hdfgroup.org/ 网页工具

使用 tools，但tools只支持六维机械臂和三个相机

## imitate_episodes.py 训练 

配置依赖 `命令行参数` 和 `data_sampler.yaml 中 的 ['act_policy']['task_config']`


```bash
CUDA_VISIBLE_DEVICES=0 python act_plus_plus/imitate_episodes.py --task_name pick_and_lift --ckpt_dir training/pick_and_lift/20demos_hdf5_4_4000_rgb --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 4 --dim_feedforward 3200 --lr 1e-5 --seed 0 --num_steps 4000

```


## mode=2 仿真测试

配置依赖  `data_sampler.yaml 中 的 ['act_policy']`

```bash
python data_sampler.py
```


# 三、其他

## 搜索 `act修改` 可以查看对 act_plus_plus 文件夹内的代码的所有改动



## rlbench 不能与 cv2 同时使用

否则会报错：

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


## 常用指令

```bash
# 在电脑和服务器间复制数据
scp -r ~/python/rlbench_imitation_learning/data/pick_and_lift/100demos_hdf5/ haoyue@100.100.3.3:/home/haoyue/code/rlbench_imitation_learning/data/pick_and_lift

scp -r -P 2122 haoyue@service.qich.top:/home/haoyue/python/rlbench_imitation_learning/training/pick_and_lift/50demosmask ~/python/rlbench_imitation_learning/training/pick_and_lift
```