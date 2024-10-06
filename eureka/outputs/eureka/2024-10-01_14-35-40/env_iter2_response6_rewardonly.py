@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis
    
    # Reward for running forward: higher speed yields a higher reward
    running_reward = forward_velocity  # Use forward velocity to encourage higher speeds
    
    # Temperature for action penalty (this can be tuned)
    temperature_action_penalty = 0.5
    action_penalty = -torch.exp(-temperature_action_penalty * torch.norm(actions, p=2, dim=-1))  # Smoothed penalty for action magnitude
    
    # Total reward combining running reward & action penalty
    total_reward = running_reward + action_penalty  # Balanced against penalties

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
