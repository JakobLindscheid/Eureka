@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_temp = 1.0
    velocity_temp = 0.05
    position_scale = 15.0  # Increased to emphasize distance penalty
    velocity_scale = -0.1   # Decreased to reduce domination of the velocity reward

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * (position_distance ** 2))  # Stronger exponential decay based on squared distance
    velocity_reward = torch.exp(velocity_temp * velocity_scale * velocity_magnitude)  # Encouraging lower velocities but less dominance

    # Total reward
    total_reward = position_reward + velocity_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
    }

    return total_reward, reward_components
