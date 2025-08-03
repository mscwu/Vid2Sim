from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from models import CustomSACPolicy, CustomPPOPolicy, CustomTD3Policy

TRIANER_DICT = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3
}

POLICY_DICT = {
    "ppo": CustomPPOPolicy,
    "sac": CustomSACPolicy,
    "td3": CustomTD3Policy
}

ENV_CLASS = {
    "dummy": DummyVecEnv,
    "subproc": SubprocVecEnv
}