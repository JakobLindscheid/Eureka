import gymnasium as gym
import torch
import numpy as np
import math
from typing import Optional, Tuple, Union, Dict, Any
from gym.spaces import Box, Discrete

class Wrapper():
    def __init__(self, env_name, cfg,  **kwargs):
        if kwargs["force_render"]:
            self.env = gym.make_vec(env_name, num_envs=1, render_mode="human")
        else:
            self.env = gym.make_vec(env_name, num_envs=cfg["env"]["numEnvs"])
        
        # because rl_games has legacy dependencies
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Box):
            self.observation_space = Box(low=obs_space.low[0], high=obs_space.high[0], shape=obs_space.shape[1:], dtype=obs_space.dtype)
        elif isinstance(obs_space, gym.spaces.MultiDiscrete):
            self.observation_space = Discrete(n=obs_space.nvec[0])
        
        act_space = self.env.action_space
        if isinstance(act_space, gym.spaces.Box):
            self.action_space = Box(low=act_space.low[0], high=act_space.high[0], shape=self.env.action_space.shape[1:], dtype=self.env.action_space.dtype)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            self.action_space = Discrete(n=act_space.nvec[0])

    def step(self, action):
        obs, gt_rew, terminated, truncated, info = self.env.step(action)

        gt_rew = torch.tensor(gt_rew)
        done = np.logical_or(terminated, truncated)
        
        self.compute_observations(obs, action, info)

        success = self.compute_success(obs, action, gt_rew, done, info)
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

        return obs, reward.numpy(), done, info # rl_games formatting (done instead of terminated/truncated)

    def reset(self):
        return self.env.reset()[0] # rl_games formatting
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def compute_success(self, obs, actions, rew, done, info):
        raise NotImplementedError
    
    def compute_observations(self, obs, actions, info):
        raise NotImplementedError
    
    def compute_reward_wrapper(self):
        return compute_reward(self.cart_position, self.cart_velocity, self.pole_angle, self.pole_velocity)
        return None, {}
    
class CartpoleGPT(Wrapper):
    def __init__(self, cfg, **kwargs):
        super().__init__("CartPole-v1", cfg, **kwargs)
        self.success = torch.zeros(self.env.num_envs)
        self.reset_buffer = np.full((self.env.num_envs), False)

    def compute_success(self, obs, actions, rew, done, info):
        self.success[self.reset_buffer] = 0
        self.success += rew
        self.reset_buffer = done
        return self.success

    def compute_observations(self, obs, actions, info):
        self.actions = torch.tensor(actions)
        self.cart_position = torch.tensor(obs[:, 0])
        self.cart_velocity = torch.tensor(obs[:, 1])
        self.pole_angle = torch.tensor(obs[:, 2])
        self.pole_velocity = torch.tensor(obs[:, 3])
def compute_reward(cart_position: torch.Tensor, 
                   cart_velocity: torch.Tensor, 
                   pole_angle: torch.Tensor, 
                   pole_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    This reward function encourages the pole to stay upright and balanced on the cart.

    The reward function provides a combination of rewards for the pole angle, pole velocity, 
    cart position, and cart velocity. Each reward component is designed to encourage the 
    desired behavior.

    :param cart_position: The position of the cart.
    :param cart_velocity: The velocity of the cart.
    :param pole_angle: The angle of the pole.
    :param pole_velocity: The velocity of the pole.
    :return: A tuple containing the total reward and a dictionary of individual reward components.
    """

    # Define the angle limit for the pole
    ANGLE_LIMIT = 0.2095  # This is the angle limit for the CartPole environment

    # Reward for staying within the angle limit
    angle_reward = 1 - (pole_angle / ANGLE_LIMIT).abs()
    # Apply transformation to normalize the reward to a fixed range
    temperature_angle = 0.5
    angle_reward = torch.exp(angle_reward / temperature_angle)

    # Reward for keeping the cart close to the center
    cart_position_reward = 1 - (cart_position / 2.4).abs()  # 2.4 is the position limit for the CartPole environment
    # Apply transformation to normalize the reward to a fixed range
    temperature_cart_position = 0.5
    cart_position_reward = torch.exp(cart_position_reward / temperature_cart_position)

    # Reward for keeping the pole velocity low
    pole_velocity_reward = 1 - (pole_velocity / 2.0).abs()  # 2.0 is an arbitrary large value
    # Apply transformation to normalize the reward to a fixed range
    temperature_pole_velocity = 0.5
    pole_velocity_reward = torch.exp(pole_velocity_reward / temperature_pole_velocity)

    # Reward for keeping the cart velocity low
    cart_velocity_reward = 1 - (cart_velocity / 2.0).abs()  # 2.0 is an arbitrary large value
    # Apply transformation to normalize the reward to a fixed range
    temperature_cart_velocity = 0.5
    cart_velocity_reward = torch.exp(cart_velocity_reward / temperature_cart_velocity)

    # Combine the rewards
    total_reward = angle_reward + cart_position_reward + pole_velocity_reward + cart_velocity_reward

    reward_components = {
        'angle_reward': angle_reward,
        'cart_position_reward': cart_position_reward,
        'pole_velocity_reward': pole_velocity_reward,
        'cart_velocity_reward': cart_velocity_reward
    }

    return total_reward, reward_components
