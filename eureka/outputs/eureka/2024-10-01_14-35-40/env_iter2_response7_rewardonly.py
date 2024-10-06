@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward direction is the x axis

    # Normalize forward velocity to be non-negative
    running_reward = torch.clip(forward_velocity, min=0.0)  # Ensure no negative rewards for backward movement
    
    # Penalty for excessive actions (to encourage efficient movement) with normalization
    action_penalty = -0.5 * (torch.norm(actions, p=2, dim=-1) ** 2)  # Halved penalty

    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
