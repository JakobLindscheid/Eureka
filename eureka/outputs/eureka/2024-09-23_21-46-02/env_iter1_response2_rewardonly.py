@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scale = 10.0  # Increased to enhance the feedback for position
    velocity_scale = -0.5   # Decreased to prevent excessive penalty
    position_temp = 0.5
    velocity_temp = 0.05

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)
    
    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * (position_distance ** 2))  # Exponential decay based on squared distance
    velocity_reward = torch.exp(-velocity_temp * velocity_scale * velocity_magnitude)  # Improve scaling and encourage lower velocities

    # Total reward
    total_reward = position_reward + velocity_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
    }

    return total_reward, reward_components
