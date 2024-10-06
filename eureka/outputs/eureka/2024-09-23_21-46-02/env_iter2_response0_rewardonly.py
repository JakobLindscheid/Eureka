@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scale = 15.0  # Increased to enhance feedback for position
    velocity_scale = -1.0   # Better penalization for high velocities
    position_temp = 0.7
    velocity_temp = 0.1
    success_radius = 0.1  # Radius around the desired target for success

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * (position_distance ** 2))  # Exponential decay based on squared distance
    stability_reward = torch.exp(velocity_temp * velocity_scale * (1 / (1 + velocity_magnitude)))  # Encourage low velocities

    # Success reward for hovering within the success radius
    success_reward = torch.where(position_distance < success_radius, torch.tensor(1.0, device=root_positions.device), torch.tensor(0.0, device=root_positions.device))

    # Total reward
    total_reward = position_reward + stability_reward + success_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'stability_reward': stability_reward,
        'success_reward': success_reward
    }

    return total_reward, reward_components
