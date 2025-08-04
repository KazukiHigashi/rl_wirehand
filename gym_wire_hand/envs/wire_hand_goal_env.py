from gymnasium import Env, spaces
from gymnasium.utils import EzPickle
import numpy as np
import mujoco
import time

from mujoco import MjModel, MjData, mj_step, mj_resetData

class WireHandGoalEnv(Env, EzPickle):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path="gym_wire_hand/envs/assets/wire_hand_damped.xml", render_mode="human"):
        EzPickle.__init__(self, xml_path, render_mode)
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)

        self.max_steps = 250
        self.step_count = 0
        self.render_mode = render_mode
        self.viewer = None

        self.action_space = spaces.Box(low=-0.35, high=0.35, shape=(self.model.nu,), dtype=np.float32)

        # Observation = dict with keys: observation, achieved_goal, desired_goal
        self.obs_dim = self.model.nq + self.model.nv + 18
        self.goal_dim = 6  # [left_goal (3), right_goal (3)]
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32)
        })

        mj_resetData(self.model, self.data)
        self._initialize_goal_template()
        self.target_r = np.zeros(3)
        self.target_l = np.zeros(3)

    def _initialize_goal_template(self):
        mj_step(self.model, self.data)
        self.palm_pos = self.data.body("wirearm").xpos.copy()
        self.radius_r = np.linalg.norm(self.data.body("right link35").xpos - self.palm_pos)
        self.radius_l = np.linalg.norm(self.data.body("left link35").xpos - self.palm_pos)
        mj_resetData(self.model, self.data)

    def _sample_random_goals(self):
        theta_r, theta_l = np.random.uniform(-np.pi / 8, np.pi / 4, size=2)
        self.target_r = self.palm_pos + self.radius_r * np.array([np.sin(theta_r), 0, -np.cos(theta_r)])
        self.target_l = self.palm_pos + self.radius_l * np.array([-np.sin(theta_l), 0, -np.cos(theta_l)])
        # optional: visualize with site_pos update
        # 目標位置の表示更新
        site_id = self.model.site("target_r").id
        self.model.site_pos[site_id] = self.target_r
        site_id = self.model.site("target_l").id
        self.model.site_pos[site_id] = self.target_l
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self.data.qpos[:] += np.random.normal(0, 0.05, size=self.data.qpos.shape)
        self._initialize_goal_template()
        self._sample_random_goals()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -0.35, 0.35)
        mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()

        info = { "is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"]) }
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        terminated = self.step_count >= self.max_steps
        return obs, reward, terminated, False, info

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos,                                         # 70
            self.data.qvel,                                         # 70
            self.data.body("left link35").xpos,                     # 3
            self.data.body("right link35").xpos,                    # 3
            self.target_l, self.target_r,                           # 6
            self.data.body("right link35").xpos - self.target_r,    # 3
            self.data.body("left link35").xpos - self.target_l      # 3
            ]).astype(np.float32)
        achieved_goal = np.concatenate([
            self.data.body("left link35").xpos,
            self.data.body("right link35").xpos
        ]).astype(np.float32)
        desired_goal = np.concatenate([self.target_l, self.target_r]).astype(np.float32)
        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }

    @staticmethod
    def _is_success(achieved_goal, desired_goal):
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)

        lp = achieved_goal[:, :3]
        rp = achieved_goal[:, 3:]
        lgoal = desired_goal[:, :3]
        rgoal = desired_goal[:, 3:]

        dist_l = np.linalg.norm(lp - lgoal, axis=1)
        dist_r = np.linalg.norm(rp - rgoal, axis=1)

        return dist_l < 0.02 and dist_r < 0.02

    def compute_reward(self, achieved_goal, desired_goal, info):
        # shape = (batch_size, 6)

        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)
        if isinstance(info, list):
            assert(isinstance(info[0], dict))
            if "e" in info[0]:
                error = np.array([a["e"] for a in info])
            else:
                error = np.zeros(len(info))
        else:
            assert(isinstance(info, dict))
            if info != {}:
                error = info["e"]
            else:
                error = 0

        lp = achieved_goal[:, :3]
        rp = achieved_goal[:, 3:]
        lgoal = desired_goal[:, :3]
        rgoal = desired_goal[:, 3:]

        dist_l = np.linalg.norm(lp - lgoal, axis=1)
        dist_r = np.linalg.norm(rp - rgoal, axis=1)

        # reward = - (dist_l + dist_r)
        reward = -2

        # シナジー（主成分空間）による再構成誤差をいい感じに0~1に変換
        # 誤差が小さいほど1に近い，シナジー計算前(errorが全部0)の場合は1と設定
        syn_rew_coeff = np.exp(-np.power(error,2)/50)

        # 成功報酬の追加（ブロードキャスト対応）
        reward += (dist_l < 0.04) * syn_rew_coeff / 2
        reward += (dist_r < 0.04) * syn_rew_coeff / 2
        reward += ((dist_l < 0.02) & (dist_r < 0.02)) * syn_rew_coeff

        return reward

    def render(self):
        if self.render_mode == "human":
            time.sleep(0.01)
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None