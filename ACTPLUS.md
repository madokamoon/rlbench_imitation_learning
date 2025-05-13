# 环境配置

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

## 训练

### sim_transfer_cube_scripted

```bash

# 生成数据集
python act_plus_plus/record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data/sim_transfer_cube_scripted --num_episodes 10 --onscreen_render



# 训练
python act_plus_plus/imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir training --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --lr 1e-5 --seed 0 --num_steps 2000
```

### pick_and_lift



训练 pick_and_lift

```bash
CUDA_VISIBLE_DEVICES=0 python act_plus_plus/imitate_episodes.py --task_name pick_and_lift --ckpt_dir training/pick_and_lift/30static_8_10000_rgb --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --lr 1e-5 --seed 0 --num_steps 10000

```




其他非必要参数

| 参数名称               | 作用                     | action           | type     | default |
|------------------------|--------------------------|------------------|----------|---------|
| --eval                 | 是否进行评估              | store_true       | -          | False   |
| --onscreen_render      | 是否进行屏幕渲染          | store_true       | -            | False   |
| --load_pretrain        | 是否加载预训练模型        | store_true       | -           | False   |
| --eval_every           | 评估间隔步数              | store            | int         | 500     |
| --validate_every       | 验证间隔步数              | store            | int         | 500     |
| --save_every           | 保存间隔步数              | store            | int         | 500     |
| --resume_ckpt_path     | 恢复训练的检查点路径      | store            | str         | -       |
| --skip_mirrored_data   | 跳过镜像数据              | store_true       | -            | False   |
| --actuator_network_dir | 执行器网络目录            | store            | str          | -       |
| --history_len          | 历史长度                  | store            | int          | -       |
| --future_len           | 未来长度                  | store            | int         | -       |
| --prediction_len       | 预测长度                  | store            | int          | -       |
| --kl_weight            | KL损失权重                | store            | int        | -       |
| --chunk_size           | 数据块大小                | store            | int          | -       |
| --hidden_dim           | 隐藏层维度                | store            | int          | -       |
| --dim_feedforward      | 前馈层维度                | store            | int        | -       |
| --temporal_agg         | 是否进行时序聚合          | store_true       | -           | False   |
| --use_vq               | 是否使用VQ               | store_true       | -            | False   |
| --vq_class             | VQ类别数量                | store            | int         | -       |
| --vq_dim               | VQ向量维度                | store            | int        | -       |
| --no_encoder           | 是否不使用编码器          | store_true       | -            | False   |



## 常用指令

```bash
# 复制训练数据
scp -r -P 2122 ~/python/rlbench_imitation_learning/data/pick_and_lift/30static_hdf5/ haoyue@service.qich.top:/home/hddData/haoyue/rlbench_imitation_learning/data/pick_and_lift
# 复制训练结果
scp -r -P 2122 haoyue@service.qich.top:/home/haoyue/python/rlbench_imitation_learning/training/pick_and_lift/50demosmask ~/python/rlbench_imitation_learning/training/pick_and_lift
```