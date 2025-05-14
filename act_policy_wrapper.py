import os
import time
from PIL import Image
import numpy as np
import time
import torch
import pickle
from einops import rearrange
import copy
# ACT模块
from act_plus_plus.utils import set_seed
from act_plus_plus.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from act_plus_plus.detr.models.latent_model import Latent_Model_Transformer


class ACTPolicyWrapper:

    def __init__(self, config):

        set_seed(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 命令行参数

        args = {
             # 原文件开头
            'eval': config["eval"],
            'ckpt_dir': config["ckpt_dir"],
            "policy_class": config['policy_class'],
            "onscreen_render": config["onscreen_render"],
            "task_name": config['task_name'],
            "batch_size": config['batch_size'],
            'num_steps': config['num_steps'],
            'eval_every': config['eval_every'],
            'validate_every': config['validate_every'],
            'save_every': config['save_every'],
            'resume_ckpt_path': config['resume_ckpt_path'],
             # 原文件 policy_config
            'lr': config['lr'],
            'num_queries': config['chunk_size'],
            'kl_weight': config['kl_weight'],
            'hidden_dim': config['hidden_dim'],
            'dim_feedforward': config['dim_feedforward'],
            'use_vq': config['use_vq'],
            'vq_class': config['vq_class'],
            'vq_dim': config['vq_dim'],
            'no_encoder': config['no_encoder'],
             # 原文件 actuator_config 
            'actuator_network_dir': config['actuator_network_dir'],
            'history_len': config['history_len'],
            'future_len': config['future_len'],
            'prediction_len': config['prediction_len'],
             # 原文件 config
            'seed': config['seed'],
            'temporal_agg': config['temporal_agg'],
            'load_pretrain': config['load_pretrain'],
            # 我的参数
            'ckpt_name': config['ckpt_name'],
        }

        is_eval = args['eval']
        ckpt_dir = args['ckpt_dir']
        policy_class = args['policy_class']
        onscreen_render = args['onscreen_render']
        task_name = args['task_name']
        batch_size_train = args['batch_size']
        batch_size_val = args['batch_size']
        num_steps = args['num_steps']
        eval_every = args['eval_every']
        validate_every = args['validate_every']
        save_every = args['save_every']
        resume_ckpt_path = args['resume_ckpt_path']

        # is_sim = task_name[:4] == 'sim_'
        # if is_sim or task_name == 'all':
        #     from act_plus.act_plus_plus.constants import SIM_TASK_CONFIGS
        #     task_config = SIM_TASK_CONFIGS[task_name]
        # else:
        #     from act_plus.act_plus_plus.constants import TASK_CONFIGS
        #     task_config = TASK_CONFIGS[task_name]
        #     task_config['dataset_dir'] = os.path.expanduser(task_config['dataset_dir'])


        task_config = config.get('task_config')

        dataset_dir = task_config['dataset_dir']
        episode_len = task_config['episode_len']
        camera_names = task_config['camera_names']
        stats_dir = task_config.get('stats_dir', None)
        sample_weights = task_config.get('sample_weights', None)
        train_ratio = task_config.get('train_ratio', 0.99)
        name_filter = task_config.get('name_filter', lambda n: True)

        # 修改维度
        state_dim = 8
        lr_backbone = 1e-5
        backbone = 'resnet18'
        if policy_class == 'ACT':
            enc_layers = 4
            dec_layers = 7
            nheads = 8
            policy_config = {'lr': args['lr'],
                            'num_queries': args['num_queries'],
                            'kl_weight': args['kl_weight'],
                            'hidden_dim': args['hidden_dim'],
                            'dim_feedforward': args['dim_feedforward'],
                            'lr_backbone': lr_backbone,
                            'backbone': backbone,
                            'enc_layers': enc_layers,
                            'dec_layers': dec_layers,
                            'nheads': nheads,
                            'camera_names': camera_names,
                            'vq': args['use_vq'],
                            'vq_class': args['vq_class'],
                            'vq_dim': args['vq_dim'],
                             # 修改维度
                            'action_dim': 10,
                            'no_encoder': args['no_encoder'],

                            # 比原版多的参数，用于 build_ACT_model_and_optimizer ，以不使用命令行参数
                            'task_name': args['task_name'],
                            'seed': args['seed'],
                            'num_steps': args["num_steps"],
                            'policy_class': args['policy_class'],
                            'ckpt_dir': args['ckpt_dir']
                            }
        else:
            raise NotImplementedError


        actuator_config = {
            'actuator_network_dir': args['actuator_network_dir'],
            'history_len': args['history_len'],
            'future_len': args['future_len'],
            'prediction_len': args['prediction_len'],
        }

        evalconfig = {
            'num_steps': num_steps,
            'eval_every': eval_every,
            'validate_every': validate_every,
            'save_every': save_every,
            'ckpt_dir': ckpt_dir,
            'resume_ckpt_path': resume_ckpt_path,
            'episode_len': episode_len,
            'state_dim': state_dim,
            'lr': args['lr'],
            'policy_class': policy_class,
            'onscreen_render': onscreen_render,
            'policy_config': policy_config,
            'task_name': task_name,
            'seed': args['seed'],
            'temporal_agg': args['temporal_agg'],
            'camera_names': camera_names,
            'real_robot': False,
            'load_pretrain': args['load_pretrain'],
            'actuator_config': actuator_config,

        }

        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        config_path = os.path.join(ckpt_dir, 'config.pkl')
        expr_name = ckpt_dir.split('/')[-1]
        
        with open(config_path, 'wb') as f:
            pickle.dump(evalconfig, f)

        # ----------------------------eval_bc 函数--------------------------------------------
        # ----------------------------eval_bc 函数--------------------------------------------

        ckpt_name = args['ckpt_name']
        save_episode=True
        num_rollouts=10

        set_seed(1000)
        ckpt_dir = evalconfig['ckpt_dir']
        state_dim = evalconfig['state_dim']
        real_robot = evalconfig['real_robot']
        policy_class = evalconfig['policy_class']
        onscreen_render = evalconfig['onscreen_render']
        policy_config = evalconfig['policy_config']
        camera_names = evalconfig['camera_names']
        max_timesteps = evalconfig['episode_len']
        task_name = evalconfig['task_name']
        temporal_agg = evalconfig['temporal_agg']
        onscreen_cam = 'angle'
        vq = evalconfig['policy_config']['vq']
        actuator_config = evalconfig['actuator_config']
        use_actuator_net = actuator_config['actuator_network_dir'] is not None

        # load policy and stats
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        policy = self.make_policy(policy_class, policy_config)
        loading_status = policy.deserialize(torch.load(ckpt_path, map_location=self.device))
        print(loading_status)
        policy.cuda()
        policy.eval()
        if vq:
            vq_dim = evalconfig['policy_config']['vq_dim']
            vq_class = evalconfig['policy_config']['vq_class']
            latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
            latent_model.deserialize(torch.load(latent_model_ckpt_path))
            latent_model.eval()
            latent_model.cuda()
            print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
        else:
            print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        if policy_class == 'Diffusion':
            self.post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        else:
            self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        # act 修改
        query_frequency = policy_config['num_queries']
        if temporal_agg:
            query_frequency = 1
            num_queries = policy_config['num_queries']
            self.num_queries = num_queries


        max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

        if temporal_agg:
            # 修改维度
            all_time_actions_zeros = torch.zeros([max_timesteps, max_timesteps+num_queries, 10]).cuda()
            self.all_time_actions_zeros = all_time_actions_zeros
            self.all_time_actions = all_time_actions_zeros
            
        # 直接创建策略实例
        self.policy = policy
        self.loading_status = loading_status
        self.temporal_agg = temporal_agg
        self.query_frequency = query_frequency
        self.camera_names = camera_names
        print(f"ACTPolicy相机名称: {self.camera_names}")
        self.step = 0

        print("ACT模型加载完成")
    

    def make_policy(self,policy_class, policy_config):
        if policy_class == 'ACT':
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        elif policy_class == 'Diffusion':
            policy = DiffusionPolicy(policy_config)
        else:
            raise NotImplementedError
        return policy


    def preprocess_images(self, imgdata):
        """预处理图像数据为模型输入格式"""
        curr_images = []
        # camera_ids = list(imgdata.keys())
        camera_ids = self.camera_names 
        print(f"preprocess_images相机ID: {camera_ids}")
        for cam_id in camera_ids:
            pil_img = Image.fromarray(imgdata[cam_id])
            resized_img = np.array(pil_img.resize((640, 480), Image.BILINEAR))
            curr_image = rearrange(resized_img, 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

        return curr_image
    

    def get_actions(self, imgdata, robot_state, view_weights=None):
        """
        从当前状态和图像预测动作
        
        参数:
            imgdata: 图像数据
            robot_state: 机器人状态
            view_weights: 各视角权重，shape为[num_cameras]，可选
        """

        with torch.inference_mode():
            time1 = time.time()

            # 处理状态数据（包括机器人状态和夹爪状态）
            state = robot_state.copy()
            qpos_numpy = np.array(state)
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            # 处理图像数据
            if self.step % self.query_frequency == 0:
                curr_image = self.preprocess_images(imgdata)

            if self.step == 0:
                # warm up
                for _ in range(10):
                    self.policy(qpos, curr_image)
                print('network warm up done')
                time1 = time.time()

            if self.step % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image, view_weights=view_weights)
                print(self.query_frequency,'次一推理')
                # print(self.all_actions.shape)
                # print(qpos)
                # print(self.all_actions[:,0:10,0:8])


            if self.temporal_agg:
                self.all_time_actions[[self.step], self.step:self.step+self.num_queries] = self.all_actions
                actions_for_curr_step = self.all_time_actions[:, self.step]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                
                # 是否开启夹爪延迟？
                # actions_for_curr_step2 = self.all_time_actions[:, max(self.step-15, 0)]
                # actions_populated2 = torch.all(actions_for_curr_step2 != 0, axis=1)
                # actions_for_curr_step2 = actions_for_curr_step2[actions_populated2]
                # k = 0.01
                # exp_weights2 = np.exp(-k * np.arange(len(actions_for_curr_step2)))
                # exp_weights2 = exp_weights2 / exp_weights2.sum()
                # exp_weights2 = torch.from_numpy(exp_weights2).cuda().unsqueeze(dim=1)
                # raw_action2 = (actions_for_curr_step2 * exp_weights2).sum(dim=0, keepdim=True)

                # 修改维度
                # raw_action[0][7]=raw_action2[0][7]

            else:
                raw_action = self.all_actions[:, self.step % self.query_frequency]

            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            target_qpos = action[:-2]

            self.step+=1

            return target_qpos

    def reset(self):
        """
        重置模型状态到初始状态，用于任务重新执行时调用
        """
        print("重置ACT策略状态...")
        # 重置步数计数器
        self.step = 0
        if self.temporal_agg:
            self.all_time_actions = self.all_time_actions_zeros
    
        print("ACT策略状态已重置")
        return True



