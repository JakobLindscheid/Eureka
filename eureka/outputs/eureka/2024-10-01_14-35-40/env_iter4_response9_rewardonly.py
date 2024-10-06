@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract torso's velocity in the x direction
    velocity = root_states[:, 7:10]  # x, y, z velocities
    forward_velocity = velocity[:, 0]  # Only the x-axis for forward movement

    # Temperature parameters for scaling
    running_temp = 0.5
    action_temp = 0.1

    # Calculate normalized running reward
    running_reward = (torch.exp(forward_velocity * running_temp) - 1) / (torch.exp(running_temp) - 1)  # Normalize to fixed range

    # Restructure action penalty to focus on efficiency (less penalty on simple actions)
    action_efficiency = -torch.norm(actions, p=2, dim=-1)  # Negative reward for using excessive control
    action_penalty = action_efficiency / (torch.norm(actions, p=2, dim=-1).clamp(min=1e-6))  # Avoid division by zero

    # Combined total reward
    total_reward = running_reward + action_penalty

    # Create reward components dictionary
    reward_components = {
        "running_reward": running_reward,
        "action_penalty": action_penalty
    }
    
    return total_reward, reward_components
