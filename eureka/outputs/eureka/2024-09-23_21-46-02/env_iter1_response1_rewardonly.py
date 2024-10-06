@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_weight = 10.0  # Increased weight to emphasize reaching the target
    velocity_weight = -0.1    # Reduced weight for velocity penalty
    position_temp = 0.05
    velocity_temp = 0.1

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity penalization
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_weight * position_distance)  # Strong incentive to get close to target
    velocity_reward = torch.exp(velocity_weight * velocity_magnitude)  # Less penalty for high velocities, reward stability

    # Total reward
    total_reward = position_reward + velocity_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
    }

    return total_reward, reward_components
