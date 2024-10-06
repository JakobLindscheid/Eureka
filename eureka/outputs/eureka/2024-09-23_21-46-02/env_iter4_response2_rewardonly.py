@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scale = 25.0  # Reduced scale for position
    velocity_scale = -5.0   # Increased penalty on undesirable speeds
    task_score_scale = 2.0   # Scale for hovering task score

    # Temperature parameters
    position_temp = 1.0
    velocity_temp = 0.1
    task_score_temp = 0.5

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * torch.clamp(position_distance, 0, 1))
    velocity_reward = torch.exp(velocity_temp * velocity_scale * torch.clamp(0.2 - velocity_magnitude, min=0.0))  # Emphasizing modest control
    task_score_reward = task_score_scale * torch.where(position_distance < 0.2, torch.tensor(1.0, device=root_positions.device), torch.tensor(0.0, device=root_positions.device))

    # Total reward
    total_reward = position_reward + velocity_reward + task_score_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
        'task_score_reward': task_score_reward,
    }

    return total_reward, reward_components
