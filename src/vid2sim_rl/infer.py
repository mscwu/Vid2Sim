import os
import random
import torch
import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import json

from env import build_env
from tqdm import trange

from omegaconf import DictConfig, OmegaConf
from static import TRIANER_DICT
from tqdm import trange

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="config", config_name="viser_demo")
def main(cfg: DictConfig):
    env_name = 'multienv' if len(cfg.env.env_paths) > 1 else os.path.basename(cfg.env.env_paths[0]).split('.')[0]
    # Create the environment
    for idx, env_path in enumerate(cfg.env.test_env_paths):
        env = build_env(env_path, cfg, worker_id=cfg.eval.work_id, random_seed=idx, inference_mode=True, no_graphics=False)
        eval_env_name = env_path.split('/')[-2]

        extra_param = {}
        if hasattr(cfg.train, 'train_freq'):
            extra_param['train_freq'] = (cfg.train.train_freq, cfg.train.train_freq_unit)

        exp_dir = os.path.join('exps', env_name, cfg.version)
        base_dir = os.path.join(exp_dir, 'eval_new', eval_env_name)

        base_dir = os.path.join(cfg.inference.output_dir, eval_env_name)
        os.makedirs(base_dir, exist_ok=True)

        if cfg.load_path.endswith('.zip'):
            ckpt_load_path = cfg.load_path
        else: 
            latest_ckpt = sorted([f for f in os.listdir(exp_dir) if f.endswith('zip')], key=lambda x: int(x.split('_')[-2]))[-1]
            ckpt_load_path = os.path.join(exp_dir, latest_ckpt)

        model = TRIANER_DICT[cfg.train.type].load(ckpt_load_path)

        print(f"Saving logs to {base_dir}")
        video_path = os.path.join(base_dir, cfg.inference.video_path)
        plot_path = os.path.join(base_dir, cfg.inference.plot_path)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        success_times = 0
        route_completion_list = []
        hit_count_list = []
        spl_list = []
        comply_rate_list = []

        for idx in trange(cfg.inference.num_episodes):
            results = {
                'distance': [],
                'reward': [],
                'step_reward': [],
            }
            acc_reward = 0
            rgb_video_writer = imageio.get_writer(os.path.join(video_path, f'episode_{idx:02d}_rgb.mp4'), fps=cfg.inference.fps)
            # depth_video_writer = imageio.get_writer(os.path.join(video_path, f'episode_{idx:02d}_depth.mp4'), fps=cfg.inference.fps)

            obs = env.reset()[0]
            done = False
            is_success = False
            step = 0
            for i in trange(cfg.inference.max_steps):
                if i % cfg.inference.decision_freq == 0:
                    action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)

                vector = obs['vector'].reshape(6,-1)[-1]
                distance = vector[-1]
                acc_reward += reward.item()

                results['distance'].append(distance)
                results['reward'].append(acc_reward)
                results['step_reward'].append(reward.item())
                
                if cfg.inference.render:
                    image = info['raw_img'][-1].transpose(1,2,0)
                    image = (image * 255).astype(np.uint8)
                    rgb_video_writer.append_data(image)
                
                    # depth = info['raw_depth'][-1][0]
                    # depth = (depth * 255).astype(np.uint8)
                    # depth_video_writer.append_data(depth)

                step += 1
                if done:
                    break

            rgb_video_writer.close()
            # depth_video_writer.close()

            start_dis = results['distance'][0]
            last_dis = results['distance'][-2]

            if abs(reward.item()-cfg.env.goal_reward) < 1:
                is_success = True
                success_times += 1

            hit_count = int(info['collision'][-2])
            lp_rate = info['collision'][-1]
            hit_count_list.append(int(hit_count))
            spl = lp_rate * int(is_success)
            spl_list.append(float(spl))
            comply_rate_list.append(float((step - hit_count) / step))
            
            route_completion = (start_dis - last_dis) / start_dis
            route_completion_list.append(float(route_completion))


        avg_route_completion = sum(route_completion_list) / cfg.inference.num_episodes
        success_rate = success_times / cfg.inference.num_episodes
        avg_hit_count = sum(hit_count_list) / cfg.inference.num_episodes
        avg_spl = sum(spl_list) / cfg.inference.num_episodes
        avg_sns = sum(comply_rate_list) / cfg.inference.num_episodes
        
        spl_var = sum([(spl - avg_spl) ** 2 for spl in spl_list]) / cfg.inference.num_episodes
        spl_std = spl_var ** 0.5

        hit_count_var = sum([(hit_count - avg_hit_count) ** 2 for hit_count in hit_count_list]) / cfg.inference.num_episodes
        hit_count_std = hit_count_var ** 0.5
        
        route_completion_var = sum([(route_completion - avg_route_completion) ** 2 for route_completion in route_completion_list]) / cfg.inference.num_episodes
        route_completion_std = route_completion_var ** 0.5

        log_dict = {
            "inference_episodes": cfg.inference.num_episodes,
            "success_rate": success_rate,
            "SPL": avg_spl,
            "route_completion": avg_route_completion,
            "hit_count": avg_hit_count,
            "compliance_rate": avg_sns,
            "SPL_std": spl_std,
            "route_completion_std": route_completion_std,
            "hit_count_std": hit_count_std,
        }

        with open(os.path.join(base_dir, 'result.json'), 'w') as f:
            json.dump(log_dict, f, indent=4)

        env.close()

if __name__ == "__main__":
    seed_everything()
    main()
