import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from gym_wire_hand.envs import WireHandGoalEnv  # 必ず学習時と同じ環境を使うこと

# ログとモデル保存パス
log_dir = "./logs"
step_n = 950000
model_path = os.path.join(log_dir, "her_sac_wirehand_{}_steps".format(step_n))
vecnormalize_path = os.path.join(log_dir, "her_sac_wirehand_vecnormalize_{}_steps.pkl".format(step_n))

# 環境を作成（Monitor → DummyVecEnv → VecNormalize に包む）
def make_env():
    return Monitor(WireHandGoalEnv())

env = DummyVecEnv([make_env])

# VecNormalize の復元（normalize を有効に）
env = VecNormalize.load(vecnormalize_path, env)
env.training = False
env.norm_reward = False  # 評価時は報酬をそのまま表示

# 学習済みモデルの読み込み
model = SAC.load(model_path, env=env)

# 1エピソードの再生
mean_reward = 0.0
for _ in range(10):
    episode_reward = 0.0
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        # 視覚化（必要に応じて）
        env.render()
    mean_reward += episode_reward
    print("Episode reward:", episode_reward)
print("Mean reward:", mean_reward/10)