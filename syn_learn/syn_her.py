from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import numpy as np
import torch as th
from typing import Optional, Dict, Any
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from syn_learn import pos_database


class HerSynergyReplayBuffer(HerReplayBuffer):
    def __init__(self, *args, pos_database=None, **kwargs):
        super().__init__(*args, **kwargs)
        if pos_database is None:
            raise ValueError("pos_database must be provided for synergy error computation")
        self.pos_database = pos_database

    def _get_real_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        obs_ = self._normalize_obs(
            {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}, env
        )
        next_obs_ = self._normalize_obs(
            {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)

        # compute PCA-based reconstruction error for real samples
        infos = [{} for _ in range(len(batch_indices))]
        pos_array = self.observations["observation"][batch_indices, env_indices, :20]  # assume first 20 dims are positions
        if len(pos_array) > 10:
            self.pos_database.calc_pca()
            rec = self.pos_database.calc_inverse(self.pos_database.calc_transform(pos_array))
            error = np.linalg.norm(pos_array - rec, axis=1)
        else:
            error = np.zeros(len(pos_array))
        for i, e in enumerate(error):
            infos[i]["e"] = e

        ag = self.next_observations["achieved_goal"][batch_indices, env_indices]
        dg = self.observations["desired_goal"][batch_indices, env_indices]

        assert self.env is not None, "HerReplayBuffer requires env for reward recomputation"
        rewards = self.env.env_method(
            "compute_reward",
            ag,
            dg,
            infos,
            indices=[0]
        )[0].astype(np.float32)

        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(
                self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),
        )