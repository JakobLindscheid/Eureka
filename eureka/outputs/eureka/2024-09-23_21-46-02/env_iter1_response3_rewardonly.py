@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scaling = 10.0
    velocity_scaling = -0.5
    position_temp = 0.1
    velocity_temp = 0.01

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity penalty
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * (position_distance / position_scaling))  # Encourage getting closer to the target
    velocity_reward = torch.exp(velocity_temp * (1.0 / (1.0 + velocity_magnitude)))  # Moderate velocity encouragement

    # Total reward
    total_reward = position_reward + velocity_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
    }

    return total_reward, reward_components
