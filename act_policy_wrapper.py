import os
import time
from PIL import Image
import numpy as np
import time
import torch
import pickle
from einops import rearrange
import hydra
import omegaconf


class ACTPolicyWrapper:

    def __init__(self, args):

        # 设置随机种子
        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])

        # 创建策略模型
        make_policy_config = omegaconf.DictConfig({
            "_target_": "act_plus_plus.detr.policy." + args['policy_class'] + ".make_policy",
            'policy_config': args
        })

        # 加载模型参数
        ckpt_dir = args['ckpt_dir']
        if args['ckpt_dir_end'] is not None:
            ckpt_dir = os.path.join(ckpt_dir, args['ckpt_dir_end'])
        ckpt_path = os.path.join(ckpt_dir, args['ckpt_name'])
        policy = hydra.utils.call(make_policy_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        loading_status = policy.deserialize(torch.load(ckpt_path, map_location=device))
        print(f'使用参数： {ckpt_path} ')
        print(f'加载情况：{loading_status}')
        policy.cuda()
        policy.eval()
        
        # vq模式
        if args['vq']:
            vq_dim = args['vq_dim']
            vq_class = args['vq_class']
            from act_plus_plus.detr.models.latent_model import Latent_Model_Transformer
            latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
            latent_model.deserialize(torch.load(latent_model_ckpt_path))
            latent_model.eval()
            latent_model.cuda()
            print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')

        # 数据集统计信息及动作处理   
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        # self参数    
        self.ckpt_path = ckpt_path
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        self.show_3D_state = args['show_3D_state']
        self.temporal_agg = args['temporal_agg']
        self.camera_names = args['camera_names']
        self.query_frequency = args['num_queries']
        self.policy = policy

        # 聚合模式
        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = args['num_queries']
            self.all_time_actions_zeros = torch.zeros([args['episode_len'],args['episode_len'] + self.num_queries, 10]).cuda()

        # 重置    
        self.reset()

        print("ACT模型初始化完成!")

    def reset(self):
        self.step = 0
        if self.temporal_agg:
            self.all_time_actions = self.all_time_actions_zeros
        return True

    def preprocess_images(self, imgdata):
        """预处理图像数据为模型输入格式"""
        curr_images = []
        # camera_ids = list(imgdata.keys())
        camera_ids = self.camera_names 
        for cam_id in camera_ids:
            pil_img = Image.fromarray(imgdata[cam_id])
            resized_img = np.array(pil_img.resize((640, 480), Image.BILINEAR))
            curr_image = rearrange(resized_img, 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image
    
    def visualize_tensor(self, tensor):
        """可视化张量中的图像"""
        import matplotlib.pyplot as plt
        
        # 转换为numpy数组
        images = tensor.cpu().numpy()[0]  # 移除批次维度
        
        plt.figure(figsize=(20, 5))
        for i in range(len(self.camera_names)):
            # 转换通道顺序 [C, H, W] -> [H, W, C]
            img = np.transpose(images[i], (1, 2, 0))
            
            plt.subplot(1, len(self.camera_names), i+1)
            plt.title(f"处理后: {self.camera_names[i]}")
            plt.imshow(img)  # 图像已归一化到[0,1]
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def get_actions(self, imgdata, robot_state, view_weights=None):
        """
        从当前状态和图像预测动作
        """

        with torch.inference_mode():
            time_start = time.time()

            # 处理状态数据（包括机器人状态和夹爪状态）
            state = robot_state.copy()
            qpos_numpy = np.array(state)
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

            # 处理图像数据
            if self.step % self.query_frequency == 0:
                curr_image = self.preprocess_images(imgdata)

                # self.visualize_tensor(curr_image)

            if self.step == 0:
                # warm up
                for _ in range(10):
                    self.policy(qpos, curr_image)

            if self.step % self.query_frequency == 0:
                if view_weights is not None:
                    self.all_actions = self.policy(qpos, curr_image, view_weights=view_weights)
                else:
                    self.all_actions = self.policy(qpos, curr_image)

                if self.show_3D_state:
                    # ------------------------------绘制---------------------------------
                    # 绘制 robot_state 和预测动作的3D可视化图
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D

                    # 创建3D图形
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    # 绘制当前机器人状态点（红色）
                    ax.scatter(robot_state[0], robot_state[1], robot_state[2], 
                            color='red', s=100, label='Robot State')

                    # 处理并绘制预测动作点（蓝色）
                    all_actions_np = self.all_actions.detach().cpu().numpy()
                    processed_actions = np.array([self.post_process(all_actions_np[0, i]) 
                                                for i in range(all_actions_np.shape[1])])
                    ax.scatter(processed_actions[:, 0], processed_actions[:, 1], processed_actions[:, 2], 
                            color='blue', alpha=0.6, label='Predicted Actions')

                    # 添加标签和标题
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'state and action (step {self.step})')
                    ax.legend()

                    max_range = 1.0
                    mid_x, mid_y, mid_z = [0.0,0.0,0.7]

                    # 设置相等的坐标轴范围
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                    ax.set_box_aspect((1, 1, 1))  # 确保坐标轴比例完全相同

                    plt.tight_layout()
                    plt.show(block=True)


                # -----------寻找最近的点 ---------------------------------
                # 计算当前状态坐标与预测动作中最近的点
                all_actions_np = self.all_actions.detach().cpu().numpy()
                # 应用post_process处理动作点
                processed_actions = np.zeros_like(all_actions_np)
                for i in range(all_actions_np.shape[1]):
                    processed_actions[0, i] = self.post_process(all_actions_np[0, i])
                
                # 获取当前状态的前3个坐标
                current_pos = robot_state[0:3]
                
                # 计算每个预测点与当前位置的欧几里得距离
                distances = np.zeros(processed_actions.shape[1])
                for i in range(processed_actions.shape[1]):
                    predicted_pos = processed_actions[0, i, 0:3]
                    distances[i] = np.linalg.norm(current_pos - predicted_pos)
                
                # 找到距离最小的点的索引
                closest_point_idx = np.argmin(distances)
                
                if 0 :   # 重新设置step为最近点的索引,这样会导致 step 变大，而超出temporal_agg模式下all_time_actions的范围，所以不能用temporal_agg
                    print(f"发现最近的点，索引从 {self.step % self.query_frequency} 调整为 {self.step + closest_point_idx}")
                    self.step = self.step + closest_point_idx
                else:    # 将closest_point_idx之前的所有动作都替换为closest_point_action ，以兼容temporal_agg模式
                    closest_point_action = self.all_actions[0, closest_point_idx].clone()
                    for i in range(closest_point_idx):
                        self.all_actions[0, i] = closest_point_action
                # ------------------------------寻找最近的点---------------------------------


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
                
                # 是否开启夹爪延迟？？ 注释为不延迟 如果修改维度 记得修改这里
                # actions_for_curr_step2 = self.all_time_actions[:, max(self.step-15, 0)]
                # actions_populated2 = torch.all(actions_for_curr_step2 != 0, axis=1)
                # actions_for_curr_step2 = actions_for_curr_step2[actions_populated2]
                # k = 0.01
                # exp_weights2 = np.exp(-k * np.arange(len(actions_for_curr_step2)))
                # exp_weights2 = exp_weights2 / exp_weights2.sum()
                # exp_weights2 = torch.from_numpy(exp_weights2).cuda().unsqueeze(dim=1)
                # raw_action2 = (actions_for_curr_step2 * exp_weights2).sum(dim=0, keepdim=True)
                # raw_action[0][7]=raw_action2[0][7]


            else:

                raw_action = self.all_actions[:, self.step % self.query_frequency]

            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            target_qpos = action[:-2]

        
            time_end = time.time()

            if self.step % self.query_frequency == 0 and self.step != 0:
                costtime = time_end - time_start
            else:
                costtime = None


            self.step+=1

            return target_qpos, costtime





