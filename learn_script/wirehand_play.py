import time
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_wire_hand.envs import WireHandEnv

import gymnasium as gym

# 環境の生成（render_mode="human" で可視化）

model_path = "logs/sac_1_20000_steps"
vec_env_path = "logs/sac_1_vecnormalize_20000_steps.pkl"

env = DummyVecEnv([lambda: WireHandEnv(render_mode="human")])
env = VecNormalize.load(vec_env_path, env)

env.training = False
env.norm_reward = True

# 学習済みモデルの読み込み
model = SAC.load(model_path)



# エピソードの実行
n_episodes = 10
for episode in range(n_episodes):
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # print(action)
        obs, reward, done, truncated = env.step(action)
        original_reward = env.get_original_reward()
        print(original_reward)
        # print(f"obs:{obs}, reward:{reward}, done:{done}, truncated{truncated}")
        env.render()
        total_reward += original_reward[0]
        step_count += 1

    print(f"Episode {episode+1}: Total Reward = {total_reward:.3f}, Steps = {step_count}")

env.close()