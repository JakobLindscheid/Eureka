@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_weight = 10.0  # Increased to enhance sensitivity to position errors
    velocity_weight = 0.01   # Reduced to lessen the penalty on velocity
    position_temp = 0.2
    velocity_temp = 0.1

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_weight * position_distance)  # Encourage getting closer to the target
    velocity_reward = -torch.exp(-velocity_temp * velocity_weight * velocity_magnitude)  # Soft penalty for velocity

    # Ensure that we provide a foundational reward for being at the target
    foundation_reward = torch.where(position_distance < 0.1, torch.tensor(1.0, device=root_positions.device), torch.tensor(0.0, device=root_positions.device))

    # Total reward
    total_reward = position_reward + velocity_reward + foundation_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
        'foundation_reward': foundation_reward,
    }

    return total_reward, reward_components
