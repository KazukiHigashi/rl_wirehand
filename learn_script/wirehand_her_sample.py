import os
import gymnasium as gym
import numpy as np

import json

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback

from syn_learn.syn_sac import SynergySAC
from gym_wire_hand.envs import WireHandGoalEnv  # GoalEnv対応済みであることが前提
from syn_learn.pos_database import SynergyManager


# JSONファイルからログ情報を読み込む
with open("config.json", "r") as f:
    config = json.load(f)

log_dir = config["log_dir"]

# チェックポイント保存設定
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=log_dir,
    name_prefix="her_sac",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# 環境のラッピング関数
def make_env():
    return Monitor(WireHandGoalEnv())

# VecNormalizeされたEnv
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

pos_database = SynergyManager(num_axis=5, init_poslist=[], maxn_pos=200)

# 普通のEnv
# env = make_env()
# env = WireHandGoalEnv()

# eval_callback = EvalCallback(env, best_model_save_path=log_dir,
#                              log_path=log_dir, eval_freq=5000,
#                              deterministic=True, render=False)

# SAC + HER のモデル設定
model = SynergySAC(
    policy="MultiInputPolicy",  # GoalEnvはdict観測なのでMultiInput
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,
    ),
    verbose=1,
    tensorboard_log="./tensorboard_log/",
    policy_kwargs=dict(net_arch=[2048, 2048, 1024]),
    batch_size=256,
    # buffer_size=8192,
    learning_starts=5000,  # エピソード終了後に十分な経験が得られる値
    # learning_rate=1e-4,
    gamma=0.95,
    tau=0.001,
    target_entropy=-30,
    train_freq=(1, "step"),
    gradient_steps=1,
    target_update_interval=2,
    pos_database=pos_database
    # ent_coef="auto_0.05"
)

# 学習の実行
model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback])

np.save(os.path.join(log_dir,"resulted_poslist.npy"), pos_database.get_npylist()[0])
np.save(os.path.join(log_dir,"resulted_targetlist.npy"), pos_database.get_npylist()[1])

# # 環境とモデルの保存
# env.save(os.path.join(log_dir, "vecnormalize.pkl"))
# model.save(os.path.join(log_dir, "her_sac_wirehand"))
