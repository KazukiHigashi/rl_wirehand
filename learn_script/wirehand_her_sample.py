import os
import gymnasium as gym
import numpy as np

import json

# from stable_baselines3 import SAC
# from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from syn_learn.syn_sac import SynergySAC
from syn_learn.syn_her import HerSynergyReplayBuffer
from syn_learn.custom_logger import configure_separate_loggers
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


pos_database = SynergyManager(num_axis=5, init_poslist=[], maxn_pos=200)
resume = True  # Trueで再学習、Falseで新規学習

logger = configure_separate_loggers(csv_folder="./csv_log", tb_folder="./tensorboard_log/")

if resume:
    # ----- 再学習 -----
    # VecNormalizeの復元
    env = DummyVecEnv([make_env])
    vecnormalize_path = os.path.join(log_dir, "her_sac_vecnormalize_600000_steps.pkl")
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = True
    env.norm_reward = True

    # 学習済みモデルのロード（envとpos_databaseを再設定）
    model_path = os.path.join(log_dir, "her_sac_600000_steps.zip")
    model = SynergySAC.load(model_path, env=env, pos_database=pos_database, replay_buffer_class=HerSynergyReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            pos_database=pos_database
        ),)

else:
    # ----- 新規学習 -----
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = SynergySAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerSynergyReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            pos_database=pos_database
        ),
        verbose=1,
        # tensorboard_log="./tensorboard_log/",
        policy_kwargs=dict(net_arch=[2048, 2048, 1024]),
        batch_size=256,
        buffer_size=2**14,
        learning_starts=5000,
        gamma=0.95,
        tau=0.001,
        target_entropy=-30,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=2,
        pos_database=pos_database
    )

model.set_logger(logger)


# 学習の実行
model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback])

np.save(os.path.join(log_dir,"resulted_poslist.npy"), pos_database.get_npylist()[0])
np.save(os.path.join(log_dir,"resulted_targetlist.npy"), pos_database.get_npylist()[1])

# # 環境とモデルの保存
# env.save(os.path.join(log_dir, "vecnormalize.pkl"))
# model.save(os.path.join(log_dir, "her_sac_wirehand"))
