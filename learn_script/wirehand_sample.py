import os

import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from gymnasium.envs.registration import register
from gym_wire_hand.envs import WireHandEnv

def make_env():
    return Monitor(WireHandEnv())

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="./logs/",
  name_prefix="sac_1",
  save_replay_buffer=True,
  save_vecnormalize=True,
)
log_dir = "./logs"

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# check_env(env)

policy_kwargs = dict(net_arch=[1024, 1024, 512])

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_log/",
            policy_kwargs=policy_kwargs, batch_size=512, target_entropy=-30.0,
            tau=0.001, gamma=0.95
            )
model.learn(total_timesteps=100_000, callback=checkpoint_callback)

env.save("vecenv_wirehand")
model.save("sac_wirehand")