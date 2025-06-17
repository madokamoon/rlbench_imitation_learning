import argparse
import yaml
import os
from data_sampler import RLBenchProcessor


def main():

    # 读取保存在training目录下的yaml文件
    parser = argparse.ArgumentParser(description='RLBench Imitation Learning')
    parser.add_argument('--ckpt', type=str, required=True)
    
    # Eval专属参数，均有默认值
    parser.add_argument('--episode_len', type=int, default=250)
    parser.add_argument('--ckpt_name', type=str, default='policy_last.ckpt')
    parser.add_argument('--show_3D_state', action='store_true', default=False)
    parser.add_argument('--show_transform_attention', action='store_true', default=False)
    parser.add_argument('--temporal_agg', action='store_true', default=False)

    args = parser.parse_args()
    
    config_path = os.path.join(args.ckpt, 'training_config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"训练配置文件未找到: {config_path}")
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # 将命令行参数添加到配置中
    cfg['policy']['ckpt_dir'] = args.ckpt
    cfg['mode'] = ["collect_and_save_demos", "process_all_epochs", "*act_eval"]
    cfg['policy']['episode_len'] = args.episode_len
    cfg['policy']['ckpt_name'] = args.ckpt_name
    cfg['policy']['show_3D_state'] = args.show_3D_state
    cfg['policy']['show_transform_attention'] = args.show_transform_attention
    cfg['policy']['temporal_agg'] = args.temporal_agg

    
    
    processor = RLBenchProcessor(cfg)
    processor.run()

if __name__ == "__main__":
    main()