@torch.jit.script
def compute_reward(root_states: torch.Tensor, prev_potentials: torch.Tensor,
                   actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # x, y, z velocities
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Reward for running forward: square of forward velocity (to emphasize higher speeds)
    running_reward = torch.square(forward_velocity) / 100.0  # Scale to [0, 1]
    running_reward = torch.clip(running_reward, min=0.0, max=1.0)  # Ensure normalization
    
    # Stronger penalty for excessive actions, clipping to avoid large penalties
    action_penalty = -torch.clip(torch.norm(actions, p=2, dim=-1) / 10.0, max=1.0)
    
    # Calculate total reward, avoiding too strong penalties that could lead to negative rewards
    total_reward = running_reward + action_penalty
    total_reward = torch.clip(total_reward, min=0.0, max=1.0)  # Normalize total reward
    
    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
