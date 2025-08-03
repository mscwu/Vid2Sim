import os
import hydra
import wandb
import time

from stable_baselines3.common.callbacks import CheckpointCallback
from utils import WandbLoggingCallback, EvalCallback
from env import auto_build_env
from static import TRIANER_DICT, POLICY_DICT, ENV_CLASS

from omegaconf import DictConfig, OmegaConf

def seed_everything(seed = 42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="config", config_name="rgb-30envs-pointnav")
def main(cfg: DictConfig):
    seed_everything(cfg.env.seed)
    # Set up training configuration
    env_name = 'multienv' if len(cfg.env.env_paths) > 1 else cfg.env.env_paths[0].split('/')[-2]
    exp_name = f'{env_name}/{cfg.version}'
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(project="Video2Sim", name=cfg.version, tags=[env_name], config=cfg_dict)
    os.makedirs(f'./exps/{exp_name}', exist_ok=True)
    OmegaConf.save(cfg, f'./exps/{exp_name}/config.yaml')

    # Create the environment (support multiple envs)
    envs, work_ids = auto_build_env(cfg.env.env_paths, cfg)
    env = ENV_CLASS[cfg.env.env_class](envs)

    extra_param = {}
    if hasattr(cfg.train, 'train_freq'):
        extra_param['train_freq'] = (cfg.train.train_freq, cfg.train.train_freq_unit)
    
    if hasattr(cfg.train, 'HER'):
        from stable_baselines3 import HerReplayBuffer
        extra_param['replay_buffer_class'] = HerReplayBuffer
        extra_param['replay_buffer_kwargs']= dict(
            n_sampled_goal=cfg.train.HER.n_sampled_goal,
            goal_selection_strategy=cfg.train.HER.goal_selection_strategy,
        )

    # Train the agent
    if cfg.resume:
        model = TRIANER_DICT[cfg.train.type].load(cfg.load_path, env=env)
        print(f"Resume model from {cfg.load_path}")
    else:
        model = TRIANER_DICT[cfg.train.type](
            POLICY_DICT[cfg.train.type], env,
            **extra_param,
            **cfg.train.trainer,
            policy_kwargs={
                'feature_extractor': cfg.train.feature_extractor,
                'net_arch': [cfg.train.net_arch.size, cfg.train.net_arch.size]
            }
        )

    if cfg.pretrained:
        model.set_parameters(cfg.load_path)
        print(f"Loaded pretrained parameters from {cfg.load_path}")

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.train.save_freq // len(cfg.env.env_paths), 
        save_path=f'./exps/{exp_name}', 
        name_prefix=cfg.version
    )
    wandb_logging_callback = WandbLoggingCallback(num_envs=len(cfg.env.env_paths))
    eval_callback = [EvalCallback(cfg, work_ids)] if cfg.eval.enabled else []
    callback_fn = [checkpoint_callback, wandb_logging_callback] + eval_callback

    model.learn(
        total_timesteps=cfg.train.total_timesteps, 
        callback=callback_fn,
        reset_num_timesteps=(not cfg.resume)
    )

    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()
