@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # x-axis corresponds to forward direction
    
    # Reward for running forward: we apply a temperature to normalize it
    temp_running = 0.5  # Temperature for running reward normalization
    running_reward = torch.exp(forward_velocity / temp_running) - 1  # Normalize to be positive
    
    # Penalty for excessive actions (with temperature adjustment)
    temp_action = 0.5  # Temperature for action penalty normalization
    action_penalty = -torch.exp(torch.norm(actions, p=2, dim=-1) / temp_action)  # Exponential penalty
    
    # Encourage longer episodes: A reward based on how long the agent has run
    episode_length_reward = torch.ones(root_states.shape[0], device=root_states.device) * 0.1  # Encourage longer episodes
    
    # Combine rewards and normalize to a fixed range
    total_reward = running_reward + action_penalty + episode_length_reward
    total_reward = torch.clip(total_reward, min=-10.0, max=10.0)
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty,
        "episode_length_reward": episode_length_reward
    }
    
    return total_reward, reward_components
