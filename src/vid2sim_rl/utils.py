import wandb
import torch
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from tqdm import trange, tqdm
from env import auto_build_env, build_env
import numpy as np
import time
from static import ENV_CLASS

class WandbLoggingCallback(BaseCallback):
    def __init__(self, num_envs=1, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.num_envs = num_envs
        self.start_dis_list = [[] for _ in range(num_envs)]
        self.cumulative_rewards = [[] for _ in range(num_envs)]
        self.entropy_list = []
        self.success_list = []

    @torch.no_grad()
    def _on_step(self) -> bool:
        for idx in range(self.num_envs):
            distance = self.locals['new_obs']['vector'][idx, -1]
            self.start_dis_list[idx].append(distance)
            self.cumulative_rewards[idx].append(self.locals["rewards"][idx])

        if hasattr(self.model.policy, 'get_distribution'):
            obs = {}
            for key, value in self.locals['new_obs'].items():
                obs[key] = torch.tensor(value).to(self.model.device)
            dist = self.model.policy.get_distribution(obs)
            entropy = dist.distribution.entropy().mean().item()
            self.entropy_list.append(entropy)

        if hasattr(self.model.policy, 'actor'):
            log_entropy_coef = self.model.log_ent_coef
            entropy = log_entropy_coef.exp().item()
            self.entropy_list.append(entropy)
                
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    c_reward = sum(self.cumulative_rewards[idx])
                    log_data = {
                        "cumulative_reward": c_reward,
                        "policy_entropy": sum(self.entropy_list) / len(self.entropy_list) if len(self.entropy_list) > 0 else None,
                    }
                    
                    start_dis = self.start_dis_list[idx][0]
                    end_dis = self.start_dis_list[idx][-2]

                    route_completion = (start_dis - end_dis) / start_dis if start_dis != 0 else 0
                    is_success = self.cumulative_rewards[idx][-1] > 5
                    self.success_list.append(is_success)
                    success_rate = sum(self.success_list) / len(self.success_list) if len(self.success_list) > 0 else 0
                    
                    collision_info = self.locals['infos'][idx]['collision']
                    log_data['Cost (Hit Count)'] = collision_info[-2]
                    log_data['Collision Rate'] = collision_info[-1] 

                    log_data['Route completion'] = route_completion
                    log_data['Success Rate'] = success_rate

                    wandb.log(log_data, step=self.num_timesteps)
                    logger.info(f"Steps: {self.num_timesteps} | Reward: {c_reward} | Hit Count: {log_data['Cost (Hit Count)']} | Collision Rate: {log_data['Collision Rate']} | Route Completion: {route_completion} | Success Rate: {success_rate}")
                    
                    self.cumulative_reward = 0
                    self.start_dis_list[idx].clear()
                    self.cumulative_rewards[idx].clear()
                    self.entropy_list = []
                    # Only keep the last 10 episodes
                    self.success_list = self.success_list[-10:]
            
        return True


class EvalCallback(BaseCallback):
    """
    Custom callback for evaluating an agent during training.

    :param eval_env_fn: Function to create the evaluation environment.
    :param eval_freq: Frequency (in steps) to perform evaluation.
    :param n_eval_episodes: Number of episodes to evaluate.
    :param inference_cfg: Configuration for inference (similar to your infer code).
    :param verbose: Verbosity level.
    """

    def __init__(self, cfg, work_ids, verbose=1):
        super(EvalCallback, self).__init__(verbose)
        self.eval_freq = cfg.eval.freq // len(cfg.env.env_paths)
        self.start_at = cfg.eval.start_at
        self.n_eval_episodes = cfg.eval.n_episodes
        self.work_ids = work_ids
        self.eval_envs, work_ids = auto_build_env(cfg.env.test_env_paths, cfg, work_ids, build_fn=build_env)
        self.cfg = cfg
        self.inference_cfg = cfg.inference
        self.current_best = 0

    def _on_step(self) -> bool:
        if self.num_timesteps < self.start_at:
            return True
        
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Starting evaluation...")
            model = self.model

            success_rate_all = 0
            route_completion_sum_all = 0.0
            avg_hit_count_all = 0.0
            avg_spl_all = 0.0

            for eval_env in tqdm(self.eval_envs):
                success_times = 0
                route_completion_sum = 0.0
                avg_hit_count = 0.0
                avg_spl = 0.0
                is_success = False
                for idx in range(self.n_eval_episodes):
                    obs, info = eval_env.reset()
                    results = {
                        'distance': [],
                        'reward': [],
                        'step_reward': [],
                    }
                    acc_reward = 0
                    done = False

                    for step in range(self.inference_cfg.max_steps):
                        if step % self.inference_cfg.decision_freq == 0:
                            action, _states = model.predict(obs, deterministic=True)
                        obs, reward, done, _, info = eval_env.step(action)

                        vector = obs['vector'].reshape(6, -1)[-1]
                        distance = vector[-1]
                        acc_reward += reward.item()

                        results['distance'].append(distance)
                        results['reward'].append(acc_reward)
                        results['step_reward'].append(reward.item())

                        if done:
                            break

                    # Compute metrics
                    start_dis = results['distance'][0] if results['distance'] else 0
                    last_dis = results['distance'][-2] if len(results['distance']) > 1 else (results['distance'][-1] if results['distance'] else 0)
                    goal_reward = self.cfg.env.goal_reward
                    if abs(reward.item() - goal_reward) < 1:
                        is_success = True
                        success_times += 1

                    collision_info = info.get('collision', [0, 0])  # Adjust based on your info structure
                    hit_count = int(collision_info[-2]) if len(collision_info) > 1 else int(collision_info[-1])
                    lp = collision_info[-1] if len(collision_info) > 0 else 0.0
                    avg_hit_count += hit_count
                    avg_spl += lp * int(is_success)

                    route_completion = (start_dis - last_dis) / start_dis if start_dis != 0 else 0
                    route_completion_sum += route_completion

                # Compute average metrics
                avg_route_completion = route_completion_sum / self.n_eval_episodes
                success_rate = success_times / self.n_eval_episodes
                avg_hit_count /= self.n_eval_episodes
                avg_spl /= self.n_eval_episodes

                if self.verbose > 1:
                    print(f"Step {self.num_timesteps}: Evaluated environment {eval_env} with success rate: {success_rate}, route completion: {avg_route_completion}, avg. hit count: {avg_hit_count}")

                success_rate_all += success_rate
                route_completion_sum_all += avg_route_completion
                avg_hit_count_all += avg_hit_count
                avg_spl_all += avg_spl

            # Compute average metrics across all evaluation environments
            success_rate_all /= len(self.eval_envs)
            route_completion_sum_all /= len(self.eval_envs)
            avg_hit_count_all /= len(self.eval_envs)
            avg_spl_all /= len(self.eval_envs)

            # Log metrics to WandB
            wandb.log({
                "eval/success_rate": success_rate_all,
                "eval/route_completion": route_completion_sum_all,
                "eval/avg_hit_count": avg_hit_count_all,
                "eval/SPL": avg_spl_all,
            }, step=self.num_timesteps)

            if success_rate_all >= self.current_best:
                self.current_best = success_rate_all
                env_name = 'multienv' if len(self.cfg.env.env_paths) > 1 else self.cfg.env.env_paths[0].split('/')[-2]
                exp_name = f'{env_name}/{self.cfg.version}'
                model.save(f"./exps/{exp_name}/best_model_{self.num_timesteps}")

            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Evaluation complete. Success rate: {success_rate_all}")

        return True