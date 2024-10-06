@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scale = 100.0  # More aggressive scaling for distance
    velocity_scale = -1.5    # Increased penalty for higher speed
    hover_penalty_scale = 20.0  # Penalizes time out of the goal region
    position_temp = 2.0
    velocity_temp = 2.0
    hover_temp = 1.0

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * position_distance)  # Decay based on distance
    velocity_reward = torch.exp(velocity_temp * velocity_scale * (velocity_magnitude - 0.2).clamp(min=0))  # Penalty on higher velocities

    # New hover based reward: Encourage staying near the target position
    hover_penalty = torch.exp(-hover_penalty_scale * position_distance.clamp(max=0.5))  # Encourage remaining close
    
    # Total reward
    total_reward = position_reward + velocity_reward + hover_penalty

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
        'hover_penalty': hover_penalty,
    }

    return total_reward, reward_components
