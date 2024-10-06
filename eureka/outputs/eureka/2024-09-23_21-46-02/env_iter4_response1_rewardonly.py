@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    
    # New scales and temperature parameters
    position_scale = 100.0  # Increased sensitivity for position
    velocity_penalty_threshold = 0.3  # Acceptable velocity
    velocity_penalty_scale = 10.0  # Penalty scaling for exceeding velocity
    task_achieved_reward = 2.0  # Additional reward for achieving task
    position_temp = 0.75
    velocity_temp = 0.1
    task_temp = 0.5

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * position_distance)  # Sensitive to proximity
    velocity_reward = torch.exp(-velocity_temp * torch.clamp(velocity_magnitude - velocity_penalty_threshold, min=0.0) * velocity_penalty_scale)  # Penalize excessive velocity

    # Reward for hovering within a small range of the target
    task_score_reward = torch.where(position_distance < 0.1, task_achieved_reward, torch.tensor(0.0, device=root_positions.device))

    # Total reward
    total_reward = position_reward + velocity_reward + task_score_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
        'task_score_reward': task_score_reward,
    }

    return total_reward, reward_components
