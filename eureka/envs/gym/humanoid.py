import gymnasium as gym
import torch
import numpy as np
import math
from typing import Optional, Tuple, Union, Dict, Any
from gym.spaces import Box

class Wrapper():
    def __init__(self, env_name, cfg,  **kwargs):
        if kwargs["force_render"]:
            self.env = gym.make_vec(env_name, num_envs=1, render_mode="human")
        else:
            self.env = gym.make_vec(env_name, num_envs=cfg["env"]["numEnvs"])
        
        # because rl_games has legacy dependencies
        obs_space = self.env.observation_space
        self.observation_space = Box(low=obs_space.low[0], high=obs_space.high[0], shape=obs_space.shape[1:], dtype=obs_space.dtype)
        act_space = self.env.action_space
        self.action_space = Box(low=act_space.low[0], high=act_space.high[0], shape=self.env.action_space.shape[1:], dtype=self.env.action_space.dtype)

    def step(self, action):
        obs, gt_rew, terminated, truncated, info = self.env.step(action)

        self.compute_observations(obs, action, info)

        success = self.compute_success(obs, action, info)
        reward, rew_info = self.compute_reward_wrapper()
        
        # if training on human reward
        if reward is None:
            reward = torch.tensor(gt_rew)
        
        info = {
            "gt_reward": gt_rew.mean(),
            "consecutive_successes": success.mean(),
            "gpt_reward": reward.mean(),
        }
        for rew_state in rew_info: info[rew_state] = rew_info[rew_state].mean()

        return obs, reward.numpy(), np.logical_or(terminated, truncated), info # rl_games formatting (done instead of terminated/truncated)

    def reset(self):
        return self.env.reset()[0] # rl_games formatting
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def compute_success(self, obs, actions, info):
        raise NotImplementedError
    
    def compute_observations(self, obs, actions, info):
        raise NotImplementedError
    
    def compute_reward_wrapper(self):
        return None, {}
    
class Humanoid(Wrapper):
    def __init__(self, cfg, **kwargs):
        super().__init__("Humanoid-v4", cfg, **kwargs)

    def compute_success(self, obs, actions, info):
        if "forward_reward" not in info:
            success = np.zeros(obs.shape[0])
        else:
            success = info["forward_reward"]
        return success

    def compute_observations(self, obs, actions, info):
        self.actions = torch.tensor(actions)
        if "x_position" not in info:
            self.x_positions= torch.zeros(obs[:, 0].shape)
        else:
            self.x_positions = torch.tensor(info["x_position"])
        if "y_position" not in info:
            self.y_positions = torch.zeros(obs[:, 0].shape)
        else:
            self.y_positions = torch.tensor(info["y_position"])
        self.z_positions = torch.tensor(obs[:, 0])
        self.x_velocities = torch.tensor(obs[:, 22])
        self.y_velocities = torch.tensor(obs[:, 23])
        self.z_velocities = torch.tensor(obs[:, 24])