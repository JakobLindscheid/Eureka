@torch.jit.script
def compute_reward(root_positions: torch.Tensor, root_linvels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define constants
    target_position = torch.tensor([0.0, 0.0, 1.0], device=root_positions.device)
    position_scale = 50.0   # Enhanced scaling for position error
    high_velocity_penalty = -3.0  # Penalty for high velocities
    hover_success_threshold = 0.5  # Define success proximity
    task_score_scale = 2.0  # Scaling for task score
    position_temp = 1.5
    velocity_temp = 0.5
    task_score_temp = 0.5

    # Calculate position error
    position_error = root_positions - target_position
    position_distance = torch.norm(position_error, dim=-1)

    # Calculate linear velocity magnitude
    velocity_magnitude = torch.norm(root_linvels, dim=-1)

    # Reward components
    position_reward = torch.exp(-position_temp * position_scale * (position_distance ** 2))  # Encourage getting closer
    velocity_penalty = high_velocity_penalty * torch.exp(velocity_temp * velocity_magnitude)  # Penalize high velocities

    # New task score based on distance to the target
    task_score = torch.where(position_distance < hover_success_threshold, 
                             torch.tensor(1.0, device=root_positions.device),
                             torch.tensor(0.0, device=root_positions.device))
    task_score_reward = task_score * task_score_scale  # Scale up the task score

    # Total reward
    total_reward = position_reward + velocity_penalty + task_score_reward

    # Create component dictionary
    reward_components = {
        'position_reward': position_reward,
        'velocity_penalty': velocity_penalty,
        'task_score_reward': task_score_reward,
    }

    return total_reward, reward_components
