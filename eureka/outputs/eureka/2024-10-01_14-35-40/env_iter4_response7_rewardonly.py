@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # Assuming the x, y, z velocities are in this slice
    forward_velocity = velocity[:, 0]  # Forward movement is along the x-axis
    
    # Calculate the running reward (more focused on forward speed)
    running_temp = 0.5
    running_reward = torch.exp(forward_velocity * running_temp) - 1  # Using exp for normalization

    # Modify action penalty to encourage efficient actions
    action_temp = 0.1
    action_penalty = -action_temp * (torch.norm(actions, p=2, dim=-1) ** 2)  # More sensitive to action size

    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
