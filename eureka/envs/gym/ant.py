import gymnasium as gym
import inspect
import torch
import numpy as np
import math
from typing import Optional, Tuple, Union, Dict, Any
from gym.spaces import Box

class AntGPT():
    def __init__(self, cfg,  **kwargs):
        if kwargs["force_render"]:
            self.env = gym.make_vec("Ant-v4", num_envs=1, render_mode="human")
        else:
            self.env = gym.make_vec("Ant-v4", num_envs=cfg["env"]["numEnvs"])
        obs_space = self.env.observation_space
        self.observation_space = Box(low=obs_space.low[0], high=obs_space.high[0], shape=obs_space.shape[1:], dtype=obs_space.dtype)
        self.action_space = Box(low=-1, high=1, shape=self.env.action_space.shape[1:], dtype=self.env.action_space.dtype)

    def step(self, action):
        obs, gt_rew, terminated, truncated, info = self.env.step(action)

        self.compute_observations(obs, action, info)

        if "reward_forward" not in info:
            success = 0
        else:
            success = info["reward_forward"]
        reward, rew_info = self.compute_reward_wrapper()
        info = {
            "gt_reward": gt_rew.mean(),
            "consecutive_successes": success.mean(),
            "gpt_reward": reward.mean(),
        }
        for rew_state in rew_info: info[rew_state] = rew_info[rew_state].mean()
        self.extras = info

        return obs, reward.numpy(), np.logical_or(terminated, truncated), info

    def reset(self):
        return self.env.reset()[0]
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
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
        self.x_velocities = torch.tensor(obs[:, 13])
        self.y_velocities = torch.tensor(obs[:, 14])
        self.z_velocities = torch.tensor(obs[:, 15])        
    
    def compute_reward_wrapper(self):
        pass