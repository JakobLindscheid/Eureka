@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scale = 50.0   # Enhanced position penalty scaling
    velocity_scale = -5.0     # Negative scaling for excessive speed
    task_score_scale = 3.0     # Scaling for ongoing success in hovering
    hover_radius = 0.3         # Radius within which success is achieved
    position_temp = 0.8
    velocity_temp = 0.1
    task_score_temp = 0.5

    # Calculate position error and distances from target
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * (position_distance ** 2))  # Encourage getting closer
    velocity_reward = torch.exp(velocity_temp * velocity_scale * velocity_magnitude)  # Penalize excessive speeds
    
    # New task score based on distance to the target and hovering context
    task_score = torch.where(position_distance < hover_radius, torch.tensor(1.0, device=root_positions.device), torch.tensor(0.0, device=root_positions.device))
    task_score_reward = task_score * task_score_scale  # Scale up the task score
    
    # Total reward
    total_reward = position_reward + velocity_reward + task_score_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_reward': velocity_reward,
        'task_score_reward': task_score_reward,
    }

    return total_reward, reward_components
