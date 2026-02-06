import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

class FrankaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # 1. Model Configuration
        model_path = os.path.join("assets", "scene.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None

        # 2. Observation Space (Total 27-dim)
        self.obs_dim = 27
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # 3. Action Space (Total 8-dim)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # 4. End-effector ID Setup
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.ee_body_id == -1:
            self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
            
        if self.ee_body_id == -1:
            raise ValueError("Target body ('hand' or 'link7') not found in the model.")

    def _get_obs(self):
        # 9 Joints: 7 Arm + 2 Gripper fingers
        qpos = self.data.qpos[:9]
        qvel = self.data.qvel[:9]
        
        # End-effector Cartesian position
        ee_pos = self.data.xpos[self.ee_body_id]
        
        # Target trajectory placeholder for future Diffusion Policy integration
        target_traj = np.zeros(6) 
        
        return np.concatenate([qpos, qvel, ee_pos, target_traj]).astype(np.float32)

    def step(self, action):
        # 1. Torque and Control Scaling
        ctrl_range = self.model.actuator_ctrlrange[:8]
        
        if np.all(ctrl_range == 0):
            applied_ctrl = action * 50.0 
        else:
            applied_ctrl = action * (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
        
        # 2. Physics Step
        self.data.ctrl[:7] = applied_ctrl[:7]
        self.data.ctrl[7:9] = applied_ctrl[7]
        
        mujoco.mj_step(self.model, self.data)
        
        # 3. State Update
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomized initial pose for generalization
        self.data.qpos[:9] += self.np_random.uniform(low=-0.05, high=0.05, size=9)
        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
