import omegaconf
import torch
import numpy as np
import os
import pickle
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
import wandb
import datetime
import pathlib
import hydra
from omegaconf import OmegaConf
from act_plus_plus.utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions

OmegaConf.register_new_resolver("eval", eval, replace=True)


import IPython
e = IPython.embed

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'act_plus_plus', 'detr', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    # 主函数，处理命令行参数并执行训练或评估
    set_seed(1)
    # 解析命令行参数
    args = cfg["policy"]
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']  # 训练时候 多少step评估一次
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']
    use_wandb = args['use_wandb']
    dataset_dir = args['dataset_dir']
    episode_len = args['episode_len']
    camera_names = args['camera_names']
    dataloader_name = args['dataloader_name']
    stats_dir = args.get('stats_dir', None)
    sample_weights = args.get('sample_weights', None)
    train_ratio = args.get('train_ratio', 0.99)
    name_filter = args.get('name_filter', lambda n: True)
    is_sim = task_name[:4] == 'sim_'
    # act修改维度
    state_dim = args['state_dim']
    lr_backbone = args['lr_backbone']
    backbone = args['backbone']

    # act修改 训练保存结果路径加入时间戳
    if  args['ckpt_dir_end'] != None:
        now_time = datetime.datetime.now()
        str_time = now_time.strftime("%Y-%m-%d-%H-%M-%S")
        ckpt_dir = os.path.join(ckpt_dir, str_time)

    # 保存配置文件到训练的目录
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f"创建目录: {ckpt_dir}")
    config_save_path = os.path.join(ckpt_dir, "training_config.yaml")
    OmegaConf.save(config=cfg, f=config_save_path, resolve=True)
    print(f"本次运行配置已保存到: {config_save_path}")

    policy_config = args
    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }

    config = {
        # 整合所有配置
        'policy_config': policy_config,
        'actuator_config': actuator_config,
    }

    if not os.path.isdir(ckpt_dir):
        # 创建检查点目录
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1]
    if not is_eval and use_wandb:
        # act修改wandb
        wandb.init(project=args['wandb_project_name'], reinit=True, name=expr_name)
        wandb.config.update(config)


    load_data_config = omegaconf.DictConfig({
        "_target_": "act_plus_plus.detr.dataloaders." + dataloader_name + ".load_data",
        'dataset_dir_l': dataset_dir,
        'camera_names': camera_names,
        'batch_size_train': batch_size_train,
        'batch_size_val': batch_size_val,
        'chunk_size': args['chunk_size'],
        'skip_mirrored_data': args['skip_mirrored_data'],
        'load_pretrain': args['load_pretrain'],
        'policy_class': policy_class,
        'stats_dir_l': stats_dir,
        'sample_weights': sample_weights,
        'train_ratio': train_ratio
    })
    train_dataloader, val_dataloader, stats, _ = hydra.utils.call(load_data_config, **{'name_filter': name_filter})

    # 保存数据集统计信息
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # 保存最佳检查点
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    if use_wandb:
        wandb.finish()

def train_bc(train_dataloader, val_dataloader, config):
    # 训练行为克隆模型
    # 读取参数
    policy_config = config['policy_config']
    num_steps = policy_config['num_steps']
    ckpt_dir = policy_config['ckpt_dir']
    seed = policy_config['seed']
    policy_class = policy_config['policy_class']
    eval_every = policy_config['eval_every']
    validate_every = policy_config['validate_every']
    save_every = policy_config['save_every']
    use_wandb = policy_config['use_wandb']
    load_pretrain = policy_config['load_pretrain']
    resume_ckpt_path = policy_config['resume_ckpt_path']
    use_weight = policy_config['use_weight']

    set_seed(seed)

    # 创建策略模型
    make_policy_config = omegaconf.DictConfig({
        "_target_": "act_plus_plus.detr.policy." + policy_class + ".make_policy",
        'policy_config': policy_config
    })
    policy = hydra.utils.call(make_policy_config)

    if load_pretrain:
        loading_status = policy.deserialize(torch.load(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'policy_step_50000_seed_0.ckpt')))
        print(f'loaded! {loading_status}')
    if resume_ckpt_path is not None:
        loading_status = policy.deserialize(torch.load(resume_ckpt_path))
        print(f'Resume policy from: {resume_ckpt_path}, Status: {loading_status}')
    policy.cuda()
    optimizer = policy.configure_optimizers()
    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps+1)):
        # validation
        # 验证
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = policy.forward_pass(data)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)
            if use_wandb:
                wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # training
        # 训练
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = policy.forward_pass(data)
        # backward
        # 反向传播
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        if use_wandb:
            wandb.log(forward_dict, step=step) # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info

def repeater(data_loader):
    # 数据加载器重复器
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    main()
