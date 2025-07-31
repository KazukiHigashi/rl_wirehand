import numpy as np
import time
import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData, mj_step, mj_resetData
from gymnasium import Env, spaces

class WireHandEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path="gym_wire_hand/envs/assets/wire_hand_damped.xml", render_mode="human"):
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)

        self.max_steps = 250
        self.step_count = 0

        self.render_mode = render_mode
        self.viewer = None

        self.action_space = spaces.Box(low=-0.35, high=0.35, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = self.model.nq + self.model.nv + 18
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        mj_resetData(self.model, self.data)
        self._initialize_goal_template()

        self.target_r = np.array([0, 0, 0])
        self.target_l = np.array([0, 0, 0])

        # Site ID 取得
        self.site_id_r = self.model.site("target_r").id
        self.site_id_l = self.model.site("target_l").id

    def _initialize_goal_template(self):
        mj_step(self.model, self.data)
        self.palm_pos = self.data.body("wirearm").xpos.copy()
        self.radius_r = np.linalg.norm(self.data.body("right link35").xpos - self.palm_pos)
        self.radius_l = np.linalg.norm(self.data.body("left link35").xpos - self.palm_pos)
        # print("l:{}, r:{}, palm:{}".format(self.radius_l, self.radius_r, self.palm_pos))
        mj_resetData(self.model, self.data)

    def _sample_random_goals(self):
        theta_r, theta_l = np.random.uniform(-np.pi/8, np.pi/4, size=2)
        self.target_r = self.palm_pos + self.radius_r * np.array([np.sin(theta_r), 0, -np.cos(theta_r)])
        self.target_l = self.palm_pos + self.radius_l * np.array([-np.sin(theta_l), 0, -np.cos(theta_l)])

        # 目標位置の表示更新
        site_id = self.model.site("target_r").id
        self.model.site_pos[site_id] = self.target_r
        site_id = self.model.site("target_l").id
        self.model.site_pos[site_id] = self.target_l
        mujoco.mj_forward(self.model, self.data)


    def reset(self, seed=None, options=None):
        mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.data.qpos[:] + np.random.normal(0, 0.05, size=self.data.qpos.shape)
        self._initialize_goal_template()
        self._sample_random_goals()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -0.35, 0.35)
        mj_step(self.model, self.data)
        self.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.step_count >= self.max_steps
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,                                         # 70
            self.data.qvel,                                         # 70
            self.data.body("left link35").xpos,                     # 3
            self.data.body("right link35").xpos,                    # 3
            self.target_l, self.target_r,                           # 6
            self.data.body("right link35").xpos - self.target_r,    # 3
            self.data.body("left link35").xpos - self.target_l      # 3
            ]).astype(np.float32)

    # def _compute_reward(self):
    #     rp = self.data.body("right link35").xpos
    #     lp = self.data.body("left link35").xpos
    #     return - (np.linalg.norm(rp - self.target_r) + np.linalg.norm(lp - self.target_l))

    def _compute_reward(self):
        rp = self.data.body("right link35").xpos
        lp = self.data.body("left link35").xpos
        dist_r = np.linalg.norm(rp - self.target_r)
        dist_l = np.linalg.norm(lp - self.target_l)

        reward = - (dist_r + dist_l)
        if dist_r < 0.02:
            reward += 0.1  # 成功報酬
        if dist_l < 0.02:
            reward += 0.1  # 成功報酬
        if dist_l < 0.02 and dist_r < 0.02:
            reward += 0.2  # 成功報酬
        return reward*10

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